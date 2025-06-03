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
    TechniqueCategory, ComplexityLevel, Industry
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
        # Find relevant papers
        relevant_papers = self.storage.find_similar_papers(user_context, n_results=max_papers)
        
        if not relevant_papers:
            return self._create_empty_recommendations()
        
        logger.info(f"Found {len(relevant_papers)} relevant papers for synthesis")
        
        # Group insights by approach
        grouped_insights = self._group_by_approach(relevant_papers)
        
        # Identify patterns
        patterns = self._identify_patterns(grouped_insights, user_context)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(patterns, user_context, grouped_insights)
        
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
            'confidence_score': self._calculate_confidence(relevant_papers, patterns)
        }
    
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
    
    def _identify_patterns(self, 
                          grouped_insights: Dict[str, List[Dict]], 
                          user_context: UserContext) -> Dict:
        """
        Identify patterns across similar implementations.
        
        Returns:
            Dict with pattern analysis results
        """
        patterns = {
            'successful_approaches': [],
            'common_success_factors': [],
            'typical_challenges': [],
            'resource_patterns': {},
            'metric_improvements': {}
        }
        
        # Analyze each group
        for approach_name, papers in grouped_insights.items():
            approach_stats = self._analyze_approach_group(papers)
            
            # Check if approach is successful
            if approach_stats['avg_quality_score'] > 0.6:
                patterns['successful_approaches'].append({
                    'name': approach_name,
                    'paper_count': len(papers),
                    'avg_quality': approach_stats['avg_quality_score'],
                    'avg_complexity': approach_stats['avg_complexity'],
                    'success_rate': approach_stats['success_rate']
                })
            
            # Extract common patterns
            patterns['common_success_factors'].extend(approach_stats['common_success_factors'])
            patterns['typical_challenges'].extend(approach_stats['common_limitations'])
            
            # Resource patterns
            if approach_stats['typical_team_size']:
                patterns['resource_patterns'][approach_name] = {
                    'team_size': approach_stats['typical_team_size'],
                    'timeline_weeks': approach_stats['avg_timeline_weeks']
                }
        
        # Deduplicate and rank patterns
        patterns['common_success_factors'] = self._deduplicate_and_rank(
            patterns['common_success_factors']
        )
        patterns['typical_challenges'] = self._deduplicate_and_rank(
            patterns['typical_challenges']
        )
        
        # Sort approaches by quality and fit
        patterns['successful_approaches'].sort(
            key=lambda x: x['avg_quality'] * (1 - self._complexity_penalty(x['avg_complexity'], user_context)),
            reverse=True
        )
        
        return patterns
    
    def _analyze_approach_group(self, papers: List[Dict]) -> Dict:
        """Analyze a group of papers using the same approach."""
        stats = {
            'avg_quality_score': 0.0,
            'avg_complexity': 0.0,
            'success_rate': 0.0,
            'common_success_factors': [],
            'common_limitations': [],
            'typical_team_size': None,
            'avg_timeline_weeks': None
        }
        
        # Collect metrics
        quality_scores = []
        complexity_levels = []
        success_factors = []
        limitations = []
        team_sizes = []
        timelines = []
        
        for paper_data in papers:
            insights = paper_data['insights']
            
            quality_scores.append(insights.get_quality_score())
            complexity_levels.append(self._complexity_to_number(insights.implementation_complexity))
            
            # Collect success factors from metrics
            for metric in insights.success_metrics:
                if metric.improvement_value and metric.improvement_value > 0:
                    success_factors.append(f"{metric.metric_name}: {metric.improvement_value}% improvement")
            
            # Collect limitations
            limitations.extend(insights.limitations[:2])  # Top 2 limitations
            
            # Resources
            if insights.resource_requirements.team_size != "not_specified":
                team_sizes.append(insights.resource_requirements.team_size)
            
            if insights.resource_requirements.estimated_time_weeks:
                timelines.append(insights.resource_requirements.estimated_time_weeks)
        
        # Calculate statistics
        if quality_scores:
            stats['avg_quality_score'] = sum(quality_scores) / len(quality_scores)
        
        if complexity_levels:
            stats['avg_complexity'] = sum(complexity_levels) / len(complexity_levels)
        
        # Success rate based on evidence strength
        high_evidence_count = sum(1 for p in papers if p['insights'].evidence_strength > 0.7)
        stats['success_rate'] = high_evidence_count / len(papers) if papers else 0
        
        # Most common patterns
        stats['common_success_factors'] = success_factors[:5]
        stats['common_limitations'] = limitations[:5]
        
        if team_sizes:
            # Most common team size
            stats['typical_team_size'] = max(set(team_sizes), key=team_sizes.count)
        
        if timelines:
            stats['avg_timeline_weeks'] = sum(timelines) / len(timelines)
        
        return stats
    
    def _generate_recommendations(self, 
                                patterns: Dict,
                                user_context: UserContext,
                                grouped_insights: Dict[str, List[Dict]]) -> Dict:
        """Generate actionable recommendations based on patterns."""
        recommendations = {
            'top_approaches': [],
            'success_factors': [],
            'common_pitfalls': [],
            'resource_requirements': {},
            'expected_outcomes': []
        }
        
        # Get top 3 approaches that fit user context
        for approach in patterns['successful_approaches'][:5]:
            approach_name = approach['name']
            
            # Check fit with user context
            if not self._approach_fits_context(approach, user_context):
                continue
            
            # Get example papers for this approach
            example_papers = grouped_insights.get(approach_name, [])[:3]
            
            recommendations['top_approaches'].append({
                'approach_name': approach_name,
                'confidence_score': approach['avg_quality'],
                'complexity': self._number_to_complexity(approach['avg_complexity']),
                'expected_timeline_weeks': patterns['resource_patterns'].get(
                    approach_name, {}
                ).get('timeline_weeks', 12),
                'team_size_required': patterns['resource_patterns'].get(
                    approach_name, {}
                ).get('team_size', 'small_team'),
                'example_implementations': [
                    {
                        'paper_id': p['paper_id'],
                        'title': self.storage.load_paper(p['paper_id']).get('title', 'Unknown'),
                        'key_insight': p['insights'].main_contribution[:100]
                    }
                    for p in example_papers
                ],
                'why_recommended': self._generate_recommendation_reason(
                    approach, user_context, example_papers
                )
            })
            
            if len(recommendations['top_approaches']) >= 3:
                break
        
        # Success factors
        recommendations['success_factors'] = self._extract_success_factors(
            patterns, user_context
        )
        
        # Common pitfalls
        recommendations['common_pitfalls'] = self._extract_pitfalls(
            patterns, grouped_insights
        )
        
        # Expected outcomes based on similar implementations
        recommendations['expected_outcomes'] = self._predict_outcomes(
            recommendations['top_approaches'], grouped_insights
        )
        
        return recommendations
    
    def _create_roadmap(self, 
                       top_approach: Dict,
                       user_context: UserContext,
                       grouped_insights: Dict[str, List[Dict]]) -> Dict:
        """Create detailed implementation roadmap."""
        approach_papers = grouped_insights.get(top_approach['approach_name'], [])
        
        # Aggregate prerequisites
        all_prerequisites = []
        for paper_data in approach_papers[:5]:
            all_prerequisites.extend(paper_data['insights'].prerequisites)
        
        # Common prerequisites
        common_prerequisites = self._deduplicate_and_rank(all_prerequisites)[:5]
        
        roadmap = {
            'approach': top_approach['approach_name'],
            'total_duration_weeks': top_approach['expected_timeline_weeks'],
            'phases': [
                {
                    'phase_number': 1,
                    'name': 'Preparation & Setup',
                    'duration_weeks': 2,
                    'activities': [
                        'Assess current infrastructure and capabilities',
                        'Identify skill gaps in team',
                        'Set up development environment',
                        'Gather required datasets or resources'
                    ],
                    'deliverables': [
                        'Gap analysis report',
                        'Environment setup complete',
                        'Initial dataset prepared'
                    ],
                    'prerequisites': common_prerequisites[:2] if common_prerequisites else []
                },
                {
                    'phase_number': 2,
                    'name': 'Prototype Development',
                    'duration_weeks': 4,
                    'activities': [
                        f'Implement basic {top_approach["approach_name"]} architecture',
                        'Develop initial evaluation framework',
                        'Create baseline measurements',
                        'Build minimal viable prototype'
                    ],
                    'deliverables': [
                        'Working prototype',
                        'Baseline performance metrics',
                        'Technical documentation'
                    ],
                    'prerequisites': common_prerequisites[2:4] if len(common_prerequisites) > 2 else []
                },
                {
                    'phase_number': 3,
                    'name': 'Iteration & Optimization',
                    'duration_weeks': 4,
                    'activities': [
                        'Refine model based on initial results',
                        'Implement performance optimizations',
                        'Conduct thorough testing',
                        'Gather stakeholder feedback'
                    ],
                    'deliverables': [
                        'Optimized implementation',
                        'Performance benchmarks',
                        'Test results documentation'
                    ]
                },
                {
                    'phase_number': 4,
                    'name': 'Production Readiness',
                    'duration_weeks': 2,
                    'activities': [
                        'Implement monitoring and logging',
                        'Create deployment pipeline',
                        'Develop operational procedures',
                        'Train support team'
                    ],
                    'deliverables': [
                        'Production-ready system',
                        'Deployment documentation',
                        'Operations manual'
                    ]
                }
            ],
            'key_milestones': [
                {'week': 2, 'milestone': 'Environment ready'},
                {'week': 6, 'milestone': 'Prototype complete'},
                {'week': 10, 'milestone': 'Optimized version ready'},
                {'week': 12, 'milestone': 'Production deployment'}
            ],
            'risk_mitigation': self._generate_risk_mitigation(top_approach, user_context)
        }
        
        return roadmap
    
    # Helper methods
    
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
    
    def _number_to_complexity(self, value: float) -> str:
        """Convert numeric complexity back to string."""
        if value <= 1.5:
            return "low"
        elif value <= 2.5:
            return "medium"
        elif value <= 3.5:
            return "high"
        else:
            return "very_high"
    
    def _complexity_penalty(self, complexity: float, user_context: UserContext) -> float:
        """Calculate penalty based on complexity vs user capabilities."""
        if user_context.maturity_level == "greenfield":
            return complexity * 0.3  # High penalty for complex approaches
        elif user_context.maturity_level == "pilot_ready":
            return complexity * 0.2
        elif user_context.maturity_level == "scaling":
            return complexity * 0.1
        else:  # optimizing
            return 0.0  # No penalty
    
    def _approach_fits_context(self, approach: Dict, user_context: UserContext) -> bool:
        """Check if an approach fits user context."""
        # Budget constraint
        if user_context.budget_constraint == "low" and approach['avg_complexity'] > 2.0:
            return False
        
        # Timeline constraint
        timeline_weeks = approach.get('timeline_weeks')
        if user_context.timeline_weeks and timeline_weeks:
            if timeline_weeks > user_context.timeline_weeks:
                return False
        
        return True
    
    def _deduplicate_and_rank(self, items: List[str]) -> List[str]:
        """Deduplicate and rank items by frequency."""
        from collections import Counter
        counter = Counter(items)
        return [item for item, _ in counter.most_common()]
    
    def _generate_recommendation_reason(self, 
                                      approach: Dict,
                                      user_context: UserContext,
                                      example_papers: List[Dict]) -> str:
        """Generate explanation for why approach is recommended."""
        reasons = []
        
        # Complexity fit
        if approach['avg_complexity'] <= 2.0:
            reasons.append("Low to medium complexity matches your team's capabilities")
        
        # Success rate
        if approach.get('success_rate', 0) > 0.7:
            reasons.append(f"High success rate ({approach['success_rate']:.0%}) in similar implementations")
        
        # Industry fit
        for paper in example_papers:
            if user_context.industry in paper['insights'].industry_applications:
                reasons.append(f"Proven in {user_context.industry.value} industry")
                break
        
        # Timeline fit
        if user_context.timeline_weeks:
            timeline = approach.get('timeline_weeks', 12)
            if timeline <= user_context.timeline_weeks:
                reasons.append(f"Can be implemented within your {user_context.timeline_weeks} week timeline")
        
        return ". ".join(reasons[:3]) if reasons else "Good fit for your requirements"
    
    def _extract_success_factors(self, patterns: Dict, user_context: UserContext) -> List[str]:
        """Extract relevant success factors."""
        factors = []
        
        # From patterns
        factors.extend(patterns.get('common_success_factors', [])[:5])
        
        # Context-specific additions
        if user_context.maturity_level == "greenfield":
            factors.append("Start with a proof of concept before full implementation")
        
        if user_context.risk_tolerance == "conservative":
            factors.append("Extensive testing and validation before production deployment")
        
        return factors[:7]  # Top 7 factors
    
    def _extract_pitfalls(self, patterns: Dict, grouped_insights: Dict) -> List[str]:
        """Extract common pitfalls to avoid."""
        pitfalls = patterns.get('typical_challenges', [])[:5]
        
        # Add generic pitfalls
        pitfalls.extend([
            "Underestimating data quality requirements",
            "Insufficient monitoring and observability",
            "Lack of stakeholder alignment"
        ])
        
        return self._deduplicate_and_rank(pitfalls)[:7]
    
    def _predict_outcomes(self, 
                         top_approaches: List[Dict],
                         grouped_insights: Dict) -> List[Dict]:
        """Predict expected outcomes based on similar implementations."""
        outcomes = []
        
        for approach in top_approaches[:1]:  # Just top approach
            approach_papers = grouped_insights.get(approach['approach_name'], [])
            
            # Aggregate metrics
            metrics_summary = defaultdict(list)
            
            for paper in approach_papers:
                for metric in paper['insights'].success_metrics:
                    if metric.improvement_value:
                        metrics_summary[metric.metric_name].append(metric.improvement_value)
            
            # Calculate averages
            for metric_name, values in metrics_summary.items():
                if values:
                    avg_improvement = sum(values) / len(values)
                    outcomes.append({
                        'metric': metric_name,
                        'expected_improvement': f"{avg_improvement:.1f}%",
                        'confidence': 'high' if len(values) >= 3 else 'medium',
                        'based_on_papers': len(values)
                    })
        
        return outcomes[:5]  # Top 5 expected outcomes
    
    def _generate_risk_mitigation(self, approach: Dict, user_context: UserContext) -> List[Dict]:
        """Generate risk mitigation strategies."""
        strategies = []
        
        # Complexity risks
        if approach.get('complexity') in ['high', 'very_high']:
            strategies.append({
                'risk': 'Implementation complexity',
                'mitigation': 'Consider phased approach with incremental milestones'
            })
        
        # Timeline risks
        if user_context.timeline_weeks and approach.get('expected_timeline_weeks', 12) > user_context.timeline_weeks * 0.8:
            strategies.append({
                'risk': 'Timeline pressure',
                'mitigation': 'Identify critical path items and parallelize where possible'
            })
        
        # Team size risks
        if user_context.company_size in ['startup', 'small']:
            strategies.append({
                'risk': 'Limited resources',
                'mitigation': 'Focus on core features first, consider external expertise for specialized tasks'
            })
        
        # Generic risks
        strategies.extend([
            {
                'risk': 'Data quality issues',
                'mitigation': 'Implement data validation pipeline early in the project'
            },
            {
                'risk': 'Stakeholder misalignment',
                'mitigation': 'Regular demos and feedback sessions throughout implementation'
            }
        ])
        
        return strategies[:5]
    
    def _calculate_confidence(self, papers: List[Dict], patterns: Dict) -> float:
        """Calculate overall confidence in recommendations."""
        factors = []
        
        # Number of papers analyzed
        if len(papers) >= 20:
            factors.append(0.9)
        elif len(papers) >= 10:
            factors.append(0.7)
        else:
            factors.append(0.5)
        
        # Quality of papers
        avg_quality = sum(p['insights'].get_quality_score() for p in papers) / len(papers) if papers else 0
        factors.append(avg_quality)
        
        # Pattern strength
        if patterns.get('successful_approaches'):
            pattern_strength = patterns['successful_approaches'][0].get('avg_quality', 0.5)
            factors.append(pattern_strength)
        
        return sum(factors) / len(factors) if factors else 0.5
    
    def _create_empty_recommendations(self) -> Dict:
        """Create empty recommendations when no relevant papers found."""
        return {
            'user_context': {},
            'papers_analyzed': 0,
            'recommendations': {
                'top_approaches': [],
                'success_factors': ['No relevant papers found for your criteria'],
                'common_pitfalls': [],
                'resource_requirements': {},
                'expected_outcomes': []
            },
            'implementation_roadmap': None,
            'confidence_score': 0.0
        }