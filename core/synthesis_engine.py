"""
Synthesis engine for creating personalized recommendations from paper insights.
Matches user contexts to relevant research and generates implementation roadmaps.
"""

import json
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import logging

from anthropic import Anthropic

from .insight_storage import InsightStorage
from .insight_schema import (
    PaperInsights, UserContext, StudyType, 
    TechniqueCategory, ComplexityLevel
)
from config import Config

logger = logging.getLogger(__name__)


class SynthesisEngine:
    """
    Creates personalized recommendations by synthesizing insights from multiple papers.
    
    Key features:
    - Pattern identification across similar implementations
    - Success factor extraction
    - Risk assessment and mitigation strategies
    - Implementation roadmap generation
    """
    
    def __init__(self, 
                 storage: Optional[InsightStorage] = None,
                 api_key: Optional[str] = None):
        """Initialize synthesis engine."""
        self.storage = storage or InsightStorage()
        self.api_key = api_key or Config.get_active_api_key()
        
        if self.api_key:
            self.llm = Anthropic(api_key=self.api_key)
        else:
            self.llm = None
            logger.warning("No API key for LLM synthesis. Using pattern-based recommendations only.")
    
    def synthesize_recommendations(self, 
                                 user_context: UserContext,
                                 max_papers: int = 50) -> Dict:
        """
        Create comprehensive recommendations based on user context.
        
        Args:
            user_context: User requirements and constraints
            max_papers: Maximum papers to analyze
            
        Returns:
            Dict with recommendations, success factors, risks, and roadmap
        """
        # Find relevant papers using enhanced vector search
        relevant_papers = self.storage.find_similar_papers(user_context, n_results=max_papers)
        
        if not relevant_papers:
            return self._create_empty_recommendations()
        
        logger.info(f"Found {len(relevant_papers)} relevant papers for synthesis")
        
        # Prioritize papers by recency, quality, evidence, and applicability
        prioritized_papers = self._prioritize_papers(relevant_papers)
        
        # Group insights by approach
        grouped_insights = self._group_by_approach(prioritized_papers)
        
        # Generate RAG-based recommendations using LLM
        if self.llm:
            recommendations = self._generate_llm_recommendations(grouped_insights, user_context)
        else:
            # Fallback to algorithmic approach
            recommendations = self._generate_algorithmic_recommendations(grouped_insights, user_context)
        
        # Create implementation roadmap for top approach
        if recommendations['top_approaches']:
            roadmap = self._create_roadmap(
                recommendations['top_approaches'][0], 
                user_context,
                grouped_insights
            )
        else:
            roadmap = None
        
        return {
            'user_context': user_context.dict(),
            'papers_analyzed': len(relevant_papers),
            'recommendations': recommendations,
            'implementation_roadmap': roadmap,
            'confidence_score': self._calculate_confidence(prioritized_papers)
        }
    
    def _prioritize_papers(self, papers: List[Dict]) -> List[Dict]:
        """
        Prioritize papers by recency, quality score, evidence strength, and practical applicability.
        """
        def priority_score(paper_data):
            insights = paper_data['insights']
            
            # Parse publication year for recency
            paper_data_full = self.storage.load_paper(paper_data['paper_id'])
            pub_year = 2020  # Default
            if paper_data_full and paper_data_full.get('published'):
                try:
                    pub_year = int(paper_data_full['published'][:4])
                except:
                    pass
            
            # Recency score (2024 = 1.0, earlier years decay)
            recency_score = max(0, 1.0 - (2024 - pub_year) * 0.1)
            
            # Combined score
            score = (
                recency_score * 0.25 +
                insights.get_quality_score() * 0.25 +
                insights.evidence_strength * 0.25 +
                insights.practical_applicability * 0.25
            )
            
            return score
        
        # Sort by priority score
        papers.sort(key=priority_score, reverse=True)
        return papers
    
    def _group_by_approach(self, papers: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group papers by their primary approach/technique.
        
        Returns:
            Dict mapping approach names to lists of papers using that approach
        """
        grouped = defaultdict(list)
        
        for paper_data in papers:
            insights = paper_data['insights']
            
            # Group by primary technique
            if insights.techniques_used:
                primary_technique = insights.techniques_used[0].value
                grouped[primary_technique].append(paper_data)
            
            # Also group by study type for empirical validation
            study_key = f"{insights.study_type.value}_studies"
            grouped[study_key].append(paper_data)
        
        # Filter out small groups
        filtered = {k: v for k, v in grouped.items() if len(v) >= 2}
        
        logger.info(f"Grouped papers into {len(filtered)} approaches")
        return filtered
    
    def _generate_llm_recommendations(self, grouped_insights: Dict[str, List[Dict]], 
                                    user_context: UserContext) -> Dict:
        """
        Generate recommendations using LLM with RAG approach.
        """
        # Prepare context from top papers
        context_papers = []
        for approach_name, papers in grouped_insights.items():
            for paper_data in papers[:3]:  # Top 3 papers per approach
                insights = paper_data['insights']
                paper_info = self.storage.load_paper(paper_data['paper_id'])
                
                # Create rich context from key findings
                key_findings_text = " ".join(insights.key_findings[:5])
                
                context_papers.append({
                    'approach': approach_name,
                    'title': paper_info.get('title', 'Unknown') if paper_info else 'Unknown',
                    'key_findings': key_findings_text,
                    'complexity': insights.implementation_complexity.value,
                    'evidence_strength': insights.evidence_strength,
                    'practical_applicability': insights.practical_applicability,
                    'techniques': [t.value for t in insights.techniques_used]
                })
        
        # Generate recommendations using LLM
        prompt = f"""Based on the following research papers about GenAI implementations, provide personalized recommendations for the user.

USER CONTEXT:
- Company Size: {user_context.company_size}
- Maturity Level: {user_context.maturity_level}  
- Budget Constraint: {user_context.budget_constraint}
- Risk Tolerance: {user_context.risk_tolerance}
- Use Case: {user_context.use_case_description}
- Preferred Techniques: {[t.value for t in user_context.preferred_techniques]}

RELEVANT RESEARCH PAPERS:
{json.dumps(context_papers[:15], indent=2)}

Generate recommendations in this JSON format:
{{
    "top_approaches": [
        {{
            "approach_name": "technique_name",
            "confidence_score": 0.85,
            "complexity": "medium",
            "why_recommended": "Specific reason based on user context and paper evidence",
            "example_implementations": [
                {{
                    "title": "Paper title",
                    "key_insight": "Most relevant finding from this paper"
                }}
            ]
        }}
    ],
    "success_factors": [
        "Factor 1 based on successful implementations in the papers",
        "Factor 2 from high-evidence research"
    ],
    "common_pitfalls": [
        "Pitfall 1 identified across multiple papers",
        "Pitfall 2 from limitation analysis"
    ],
    "expected_outcomes": [
        "Outcome 1 with quantitative backing from research",
        "Outcome 2 based on similar implementations"
    ]
}}

Focus on:
1. Matching approaches to user constraints (budget, risk, complexity)
2. Prioritizing recent, high-evidence research
3. Extracting actionable insights from key findings
4. Providing specific, evidence-based recommendations"""

        try:
            response = self.llm.messages.create(
                model=Config.LLM_MODEL,
                temperature=0.3,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result = json.loads(response.content[0].text)
            return result
            
        except Exception as e:
            logger.error(f"LLM recommendation generation failed: {e}")
            # Fallback to algorithmic approach
            return self._generate_algorithmic_recommendations(grouped_insights, user_context)
    
    def _generate_algorithmic_recommendations(self, grouped_insights: Dict[str, List[Dict]], 
                                           user_context: UserContext) -> Dict:
        """
        Fallback algorithmic recommendation generation.
        """
        recommendations = {
            'top_approaches': [],
            'success_factors': [],
            'common_pitfalls': [],
            'expected_outcomes': []
        }
        
        # Analyze each approach
        approach_scores = []
        for approach_name, papers in grouped_insights.items():
            if approach_name.endswith('_studies'):
                continue  # Skip study type groupings
            
            # Calculate approach score
            avg_quality = sum(p['insights'].get_quality_score() for p in papers) / len(papers)
            avg_evidence = sum(p['insights'].evidence_strength for p in papers) / len(papers)
            avg_applicability = sum(p['insights'].practical_applicability for p in papers) / len(papers)
            
            # Complexity penalty based on user constraints
            complexity_penalty = self._calculate_complexity_penalty(papers, user_context)
            
            overall_score = (avg_quality + avg_evidence + avg_applicability) / 3 - complexity_penalty
            
            approach_scores.append({
                'approach_name': approach_name,
                'score': overall_score,
                'confidence_score': avg_evidence,
                'complexity': self._get_average_complexity(papers),
                'paper_count': len(papers),
                'papers': papers[:3]  # Top 3 examples
            })
        
        # Sort by score and take top approaches
        approach_scores.sort(key=lambda x: x['score'], reverse=True)
        
        for approach in approach_scores[:3]:
            recommendations['top_approaches'].append({
                'approach_name': approach['approach_name'],
                'confidence_score': approach['confidence_score'],
                'complexity': approach['complexity'],
                'why_recommended': self._generate_recommendation_reason(approach, user_context),
                'example_implementations': [
                    {
                        'title': self.storage.load_paper(p['paper_id']).get('title', 'Unknown'),
                        'key_insight': p['insights'].key_findings[0] if p['insights'].key_findings else 'No key insight available'
                    }
                    for p in approach['papers']
                ]
            })
        
        # Extract success factors and pitfalls
        all_findings = []
        all_limitations = []
        
        for papers in grouped_insights.values():
            for paper_data in papers:
                insights = paper_data['insights']
                all_findings.extend(insights.key_findings)
                all_limitations.extend(insights.limitations)
        
        # Simple frequency-based extraction
        recommendations['success_factors'] = self._extract_common_themes(all_findings)[:5]
        recommendations['common_pitfalls'] = self._extract_common_themes(all_limitations)[:5]
        recommendations['expected_outcomes'] = ["Improved efficiency", "Reduced manual effort", "Enhanced accuracy"]
        
        return recommendations
    
    def _calculate_complexity_penalty(self, papers: List[Dict], user_context: UserContext) -> float:
        """Calculate penalty based on complexity vs user capabilities."""
        avg_complexity = sum(self._complexity_to_number(p['insights'].implementation_complexity) 
                           for p in papers) / len(papers)
        
        if user_context.budget_constraint == "low":
            return avg_complexity * 0.3
        elif user_context.budget_constraint == "medium":
            return avg_complexity * 0.2
        else:
            return 0.0
    
    def _complexity_to_number(self, complexity: ComplexityLevel) -> float:
        """Convert complexity level to numeric value."""
        mapping = {
            ComplexityLevel.LOW: 1.0,
            ComplexityLevel.MEDIUM: 2.0,
            ComplexityLevel.HIGH: 3.0,
            ComplexityLevel.VERY_HIGH: 4.0,
            ComplexityLevel.UNKNOWN: 2.5
        }
        return mapping.get(complexity, 2.5)
    
    def _get_average_complexity(self, papers: List[Dict]) -> str:
        """Get average complexity level for a group of papers."""
        avg_num = sum(self._complexity_to_number(p['insights'].implementation_complexity) 
                     for p in papers) / len(papers)
        
        if avg_num <= 1.5:
            return "low"
        elif avg_num <= 2.5:
            return "medium"
        elif avg_num <= 3.5:
            return "high"
        else:
            return "very_high"
    
    def _generate_recommendation_reason(self, approach: Dict, user_context: UserContext) -> str:
        """Generate explanation for why approach is recommended."""
        reasons = []
        
        # Complexity fit
        if approach['score'] > 0.7:
            reasons.append("High success rate in similar implementations")
        
        # Evidence backing
        if approach['confidence_score'] > 0.7:
            reasons.append("Strong empirical evidence")
        
        # Budget fit
        if approach['complexity'] in ['low', 'medium'] and user_context.budget_constraint in ['low', 'medium']:
            reasons.append("Matches your budget constraints")
        
        # Risk fit
        if user_context.risk_tolerance == "conservative" and approach['confidence_score'] > 0.6:
            reasons.append("Low-risk approach with proven results")
        
        return ". ".join(reasons[:3]) if reasons else "Good fit for your requirements based on research analysis"
    
    def _extract_common_themes(self, text_list: List[str]) -> List[str]:
        """Extract common themes from a list of text strings."""
        # Simple keyword-based theme extraction
        common_words = {}
        
        for text in text_list:
            words = text.lower().split()
            for word in words:
                if len(word) > 4:  # Skip short words
                    common_words[word] = common_words.get(word, 0) + 1
        
        # Return most common meaningful phrases
        sorted_words = sorted(common_words.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:10] if count > 1]
    
    def _create_roadmap(self, top_approach: Dict, user_context: UserContext, 
                       grouped_insights: Dict[str, List[Dict]]) -> Dict:
        """Create detailed implementation roadmap."""
        approach_papers = grouped_insights.get(top_approach['approach_name'], [])
        
        # Aggregate prerequisites
        all_prerequisites = []
        for paper_data in approach_papers[:5]:
            all_prerequisites.extend(paper_data['insights'].prerequisites)
        
        # Common prerequisites
        common_prerequisites = list(set(all_prerequisites))[:5]
        
        roadmap = {
            'approach': top_approach['approach_name'],
            'total_duration_weeks': 12,  # Default timeline
            'phases': [
                {
                    'phase_number': 1,
                    'name': 'Research & Planning',
                    'duration_weeks': 2,
                    'activities': [
                        'Review relevant research papers and implementations',
                        'Assess current infrastructure and team capabilities',
                        'Identify specific requirements and constraints',
                        'Create detailed project plan and timeline'
                    ],
                    'deliverables': [
                        'Research analysis report',
                        'Technical requirements document',
                        'Project roadmap and timeline'
                    ],
                    'prerequisites': common_prerequisites[:2] if common_prerequisites else []
                },
                {
                    'phase_number': 2,
                    'name': 'Prototype Development',
                    'duration_weeks': 4,
                    'activities': [
                        f'Implement basic {top_approach["approach_name"]} prototype',
                        'Set up evaluation framework and metrics',
                        'Create initial test datasets',
                        'Develop baseline performance measurements'
                    ],
                    'deliverables': [
                        'Working prototype',
                        'Evaluation framework',
                        'Baseline performance metrics'
                    ],
                    'prerequisites': common_prerequisites[2:4] if len(common_prerequisites) > 2 else []
                },
                {
                    'phase_number': 3,
                    'name': 'Optimization & Testing',
                    'duration_weeks': 4,
                    'activities': [
                        'Refine implementation based on initial results',
                        'Conduct comprehensive testing and validation',
                        'Optimize performance and resource usage',
                        'Document lessons learned and best practices'
                    ],
                    'deliverables': [
                        'Optimized implementation',
                        'Test results and validation report',
                        'Performance optimization guide'
                    ]
                },
                {
                    'phase_number': 4,
                    'name': 'Deployment & Monitoring',
                    'duration_weeks': 2,
                    'activities': [
                        'Prepare production deployment pipeline',
                        'Implement monitoring and alerting systems',
                        'Create operational procedures and documentation',
                        'Train team on maintenance and troubleshooting'
                    ],
                    'deliverables': [
                        'Production-ready system',
                        'Monitoring dashboard',
                        'Operations manual and training materials'
                    ]
                }
            ],
            'key_milestones': [
                {'week': 2, 'milestone': 'Research complete, plan approved'},
                {'week': 6, 'milestone': 'Working prototype demonstrated'},
                {'week': 10, 'milestone': 'Optimized version ready for deployment'},
                {'week': 12, 'milestone': 'System deployed and operational'}
            ],
            'risk_mitigation': self._generate_risk_mitigation(top_approach, user_context)
        }
        
        return roadmap
    
    def _generate_risk_mitigation(self, approach: Dict, user_context: UserContext) -> List[Dict]:
        """Generate risk mitigation strategies."""
        strategies = []
        
        # Complexity risks
        if approach.get('complexity') in ['high', 'very_high']:
            strategies.append({
                'risk': 'Implementation complexity',
                'mitigation': 'Break down into smaller, manageable phases with regular checkpoints'
            })
        
        # Budget risks
        if user_context.budget_constraint in ['low', 'medium']:
            strategies.append({
                'risk': 'Budget constraints',
                'mitigation': 'Focus on core functionality first, consider open-source alternatives'
            })
        
        # Team capability risks
        if user_context.company_size in ['startup', 'small']:
            strategies.append({
                'risk': 'Limited team expertise',
                'mitigation': 'Invest in training early, consider external consulting for critical phases'
            })
        
        # Generic risks
        strategies.extend([
            {
                'risk': 'Technology maturity',
                'mitigation': 'Stay updated with latest research, have fallback approaches ready'
            },
            {
                'risk': 'Stakeholder alignment',
                'mitigation': 'Regular demos and progress reviews, clear success metrics'
            }
        ])
        
        return strategies[:5]
    
    def _calculate_confidence(self, papers: List[Dict]) -> float:
        """Calculate overall confidence in recommendations."""
        if not papers:
            return 0.0
        
        # Average quality and evidence strength
        avg_quality = sum(p['insights'].get_quality_score() for p in papers) / len(papers)
        avg_evidence = sum(p['insights'].evidence_strength for p in papers) / len(papers)
        
        # Paper count factor
        count_factor = min(1.0, len(papers) / 20.0)  # Full confidence at 20+ papers
        
        return (avg_quality * 0.4 + avg_evidence * 0.4 + count_factor * 0.2)
    
    def _create_empty_recommendations(self) -> Dict:
        """Create empty recommendations when no relevant papers found."""
        return {
            'user_context': {},
            'papers_analyzed': 0,
            'recommendations': {
                'top_approaches': [],
                'success_factors': ['No relevant papers found for your criteria'],
                'common_pitfalls': [],
                'expected_outcomes': []
            },
            'implementation_roadmap': None,
            'confidence_score': 0.0
        }