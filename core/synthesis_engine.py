"""
Synthesis engine for creating personalized recommendations from paper insights.
Uses a consultant-style LLM approach to match user contexts to relevant research.
"""

import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
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
    Creates personalized recommendations using a consultant-style LLM approach.
    
    Emulates McKinsey/Deloitte-style consulting to:
    - Understand client business context
    - Analyze research corpus
    - Recommend best-fit solutions
    - Provide strategic implementation guidance
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
            raise ValueError("LLM API key required for synthesis engine")
    
    def synthesize_recommendations(self, 
                                 user_context: UserContext,
                                 max_papers: int = 50,
                                 interactive: bool = False) -> Dict:
        """
        Create consultant-grade recommendations based on user context.
        
        Args:
            user_context: User requirements and constraints
            max_papers: Maximum papers to retrieve initially
            interactive: Whether to enable interactive follow-up
            
        Returns:
            Dict with recommendations and strategic insights
        """
        # Step 1: Retrieve relevant papers using vector search
        relevant_papers = self.storage.find_similar_papers(user_context, n_results=max_papers)
        
        if not relevant_papers:
            return self._create_empty_recommendations()
        
        logger.info(f"Retrieved {len(relevant_papers)} relevant papers for analysis")
        
        # Step 2: Rerank papers using quality metrics
        reranked_papers = self._rerank_papers(relevant_papers)
        
        # Step 3: Select top papers for synthesis (cap at 25 for context window)
        top_papers = reranked_papers[:25]
        
        # Step 4: Generate consultant-style synthesis
        synthesis_result = self._generate_consultant_synthesis(
            top_papers, 
            user_context,
            interactive
        )
        
        return synthesis_result
    
    def _rerank_papers(self, papers: List[Dict]) -> List[Dict]:
        """
        Rerank papers using the consultant-specified formula:
        final_score = 0.4 * quality + 0.3 * evidence + 0.2 * applicability + 0.1 * recency
        """
        current_year = datetime.now().year
        
        for paper_data in papers:
            insights = paper_data['insights']
            
            # Get publication year for recency calculation
            paper_info = self.storage.load_paper(paper_data['paper_id'])
            pub_year = 2020  # Default
            if paper_info and paper_info.get('published'):
                try:
                    pub_year = int(paper_info['published'][:4])
                except:
                    pass
            
            # Calculate recency score (normalize to 0-1)
            years_old = current_year - pub_year
            recency_score = max(0, 1 - (years_old * 0.1))  # 10% decay per year
            
            # Calculate final reranking score
            paper_data['rerank_score'] = (
                0.4 * insights.get_quality_score() +
                0.3 * insights.evidence_strength +
                0.2 * insights.practical_applicability +
                0.1 * recency_score
            )
            
            # Store components for transparency
            paper_data['score_components'] = {
                'quality': insights.get_quality_score(),
                'evidence': insights.evidence_strength,
                'applicability': insights.practical_applicability,
                'recency': recency_score
            }
        
        # Sort by rerank score
        papers.sort(key=lambda x: x['rerank_score'], reverse=True)
        return papers
    
    def _generate_consultant_synthesis(self, 
                                     papers: List[Dict], 
                                     user_context: UserContext,
                                     interactive: bool = False) -> Dict:
        """
        Generate McKinsey-style synthesis using LLM.
        """
        # Prepare research summaries
        research_summaries = self._prepare_research_summaries(papers)
        
        # Create consultant prompt
        prompt = self._create_consultant_prompt(user_context, research_summaries, interactive)
        
        try:
            response = self.llm.messages.create(
                model=Config.LLM_MODEL,
                temperature=0.3,  # Lower temperature for professional consistency
                max_tokens=3000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse the structured response
            synthesis = self._parse_consultant_response(response.content[0].text)
            
            # Add metadata
            synthesis['metadata'] = {
                'papers_analyzed': len(papers),
                'synthesis_timestamp': datetime.utcnow().isoformat(),
                'top_paper_scores': [
                    {
                        'title': self.storage.load_paper(p['paper_id']).get('title', 'Unknown')[:100],
                        'score': p['rerank_score'],
                        'components': p['score_components']
                    }
                    for p in papers[:5]
                ]
            }
            
            return synthesis
            
        except Exception as e:
            logger.error(f"Consultant synthesis generation failed: {e}")
            return self._create_fallback_synthesis(papers, user_context)
    
    def _prepare_research_summaries(self, papers: List[Dict]) -> List[Dict]:
        """
        Prepare research summaries in consultant-friendly format.
        """
        summaries = []
        
        for i, paper_data in enumerate(papers):
            insights = paper_data['insights']
            paper_info = self.storage.load_paper(paper_data['paper_id'])
            
            if not paper_info:
                continue
            
            # Create 200-word key findings summary
            key_findings_text = self._create_findings_summary(insights)
            
            # Identify primary approach/technique
            primary_approach = "General AI approach"
            if insights.techniques_used:
                primary_approach = insights.techniques_used[0].value.replace('_', ' ').title()
            
            summary = {
                'rank': i + 1,
                'title': paper_info.get('title', 'Unknown'),
                'approach': primary_approach,
                'findings': key_findings_text,
                'limitations': '; '.join(insights.limitations[:2]) if insights.limitations else "Not specified",
                'quality_score': insights.get_quality_score(),
                'evidence_strength': insights.evidence_strength,
                'practical_applicability': insights.practical_applicability,
                'complexity': insights.implementation_complexity.value,
                'published_year': self._extract_year(paper_info)
            }
            
            summaries.append(summary)
        
        return summaries
    
    def _create_findings_summary(self, insights: PaperInsights) -> str:
        """
        Create a concise 200-word summary of key findings and conclusions.
        """
        # Combine key findings
        findings_text = " ".join(insights.key_findings[:3])  # Top 3 findings
        
        # Add problem addressed if available
        if insights.problem_addressed:
            findings_text = f"Addresses: {insights.problem_addressed}. {findings_text}"
        
        # Add success metrics if available
        if insights.success_metrics:
            metrics_text = ", ".join([
                f"{m.metric_name}: {m.improvement_value}{m.improvement_unit or ''}"
                for m in insights.success_metrics[:2]
            ])
            if metrics_text:
                findings_text += f" Achieved: {metrics_text}."
        
        # Truncate to ~200 words (roughly 1000 characters)
        if len(findings_text) > 1000:
            findings_text = findings_text[:997] + "..."
        
        return findings_text
    
    def _create_consultant_prompt(self, 
                                user_context: UserContext, 
                                research_summaries: List[Dict],
                                interactive: bool) -> str:
        """
        Create McKinsey-style consultant prompt.
        """
        # Format user context
        context_details = []
        if user_context.company_size:
            context_details.append(f"Company Size: {user_context.company_size}")
        if user_context.maturity_level:
            context_details.append(f"AI Maturity: {user_context.maturity_level}")
        if user_context.budget_constraint:
            context_details.append(f"Budget Level: {user_context.budget_constraint}")
        if user_context.use_case_description:
            context_details.append(f"Business Context: {user_context.use_case_description}")
        if user_context.specific_problems:
            context_details.append(f"Specific Challenges: {', '.join(user_context.specific_problems)}")
        
        context_text = "\n".join(context_details) if context_details else "General exploration of GenAI applications"
        
        # Format research summaries
        summaries_text = ""
        for summary in research_summaries:
            summaries_text += f"""
{summary['rank']}. Title: {summary['title']}
   Approach: {summary['approach']}
   Key Findings: {summary['findings']}
   Limitations: {summary['limitations']}
   Scores: Quality={summary['quality_score']:.2f}, Evidence={summary['evidence_strength']:.2f}, Applicability={summary['practical_applicability']:.2f}
   Complexity: {summary['complexity']} | Year: {summary['published_year']}
"""

        # Create consultant prompt
        prompt = f"""You are a senior consultant at McKinsey & Company specializing in AI strategy and implementation. Your client has approached you for guidance on implementing generative AI solutions.

## Client Context
{context_text}

## Research Analysis
I've analyzed {len(research_summaries)} relevant research papers and implementations. Here are the top findings:

{summaries_text}

## Your Consulting Tasks

Please provide a strategic recommendation following this structure:

1. **Executive Summary** (2-3 sentences)
   - What is your primary recommendation?
   - Why is this the best fit for the client?

2. **Recommended Approach**
   - Primary technique/methodology recommended
   - Why this approach aligns with client constraints
   - Expected complexity and resource requirements

3. **Strategic Advantages**
   - 3-4 key benefits specific to the client's context
   - Competitive advantages this approach provides
   - ROI indicators based on research evidence

4. **Risk Assessment**
   - 2-3 primary risks or challenges
   - Mitigation strategies for each risk
   - Critical success factors

5. **Implementation Considerations**
   - High-level phases (do not detail full roadmap yet)
   - Key prerequisites or dependencies
   - Recommended team composition

{"6. **Next Steps**" if interactive else "6. **Getting Started**"}
{" - Would you like me to develop a detailed implementation roadmap?" if interactive else " - Key actions for the first 30 days"}
{" - Are there alternative approaches you'd like to explore?" if interactive else " - Quick wins to build momentum"}
{" - Do you have specific concerns about this recommendation?" if interactive else " - Success metrics to track"}

Please structure your response in a clear, professional format using markdown. Be specific and actionable, referencing the research evidence where appropriate. Maintain a consultative tone that builds confidence while being transparent about challenges."""

        return prompt
    
    def _parse_consultant_response(self, response_text: str) -> Dict:
        """
        Parse the consultant response into structured format.
        """
        # For now, return the response as-is with basic structure
        # In production, this could parse specific sections
        return {
            'consultant_analysis': response_text,
            'recommendations': {
                'full_text': response_text,
                'interactive_options': [
                    "Generate detailed implementation roadmap",
                    "Explore alternative approaches",
                    "Deep dive on specific risks",
                    "Analyze cost-benefit in detail"
                ]
            }
        }
    
    def generate_implementation_roadmap(self, 
                                      previous_synthesis: Dict,
                                      user_context: UserContext) -> Dict:
        """
        Generate detailed implementation roadmap based on previous synthesis.
        """
        prompt = f"""As the McKinsey consultant who provided the previous recommendation, the client has now asked for a detailed implementation roadmap.

## Previous Recommendation Summary
{previous_synthesis.get('consultant_analysis', 'Previous recommendation not found')}

## Client Context (Reminder)
Company Size: {user_context.company_size}
Budget: {user_context.budget_constraint}

Please provide a detailed implementation roadmap with:

1. **Phase-by-Phase Plan** (align with client's timeline)
   - Phase name and duration
   - Key activities and deliverables
   - Success criteria
   - Dependencies and prerequisites

2. **Resource Plan**
   - Team structure and roles needed
   - Estimated effort by role
   - External resources or vendors required
   - Budget allocation guidance

3. **Risk Management Plan**
   - Risk register with probability and impact
   - Mitigation strategies by phase
   - Decision gates and checkpoints

4. **Success Metrics & KPIs**
   - Leading indicators by phase
   - Lagging indicators for overall success
   - Measurement methodology
   - Reporting cadence

5. **Change Management**
   - Stakeholder engagement plan
   - Communication strategy
   - Training requirements
   - Adoption strategies

Format as a professional consulting deliverable with clear sections and actionable details."""

        try:
            response = self.llm.messages.create(
                model=Config.LLM_MODEL,
                temperature=0.3,
                max_tokens=3000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return {
                'implementation_roadmap': response.content[0].text,
                'roadmap_generated': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Roadmap generation failed: {e}")
            return {'error': 'Failed to generate roadmap', 'details': str(e)}
    
    def explore_alternative_approaches(self,
                                     papers: List[Dict],
                                     user_context: UserContext,
                                     num_alternatives: int = 3) -> Dict:
        """
        Explore alternative approaches based on different criteria.
        """
        prompt = f"""As the McKinsey consultant, the client has asked to explore alternative approaches beyond the primary recommendation.

## Client Context
{self._format_user_context(user_context)}

## Available Approaches from Research
{self._format_approaches_summary(papers)}

Please provide {num_alternatives} alternative approaches, each optimized for different priorities:

For each alternative, provide:
1. **Approach Name & Rationale**
   - What priority does this optimize for? (e.g., speed, cost, risk, innovation)
   - Key differentiator from primary recommendation

2. **Pros and Cons**
   - Specific advantages for this client
   - Trade-offs compared to primary recommendation

3. **Best Fit Scenario**
   - When to choose this approach
   - Conditions for success

4. **Implementation Complexity**
   - Relative to primary recommendation
   - Key challenges to expect

Format as a comparison matrix that helps the client make an informed decision."""

        try:
            response = self.llm.messages.create(
                model=Config.LLM_MODEL,
                temperature=0.4,  # Slightly higher for creative alternatives
                max_tokens=2500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return {
                'alternative_approaches': response.content[0].text,
                'alternatives_generated': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Alternative exploration failed: {e}")
            return {'error': 'Failed to generate alternatives', 'details': str(e)}
    
    def _format_user_context(self, user_context: UserContext) -> str:
        """Format user context for prompts."""
        parts = []
        if user_context.company_size:
            parts.append(f"Company Size: {user_context.company_size}")
        if user_context.use_case_description:
            parts.append(f"Use Case: {user_context.use_case_description}")
        if user_context.budget_constraint:
            parts.append(f"Budget: {user_context.budget_constraint}")
        return "\n".join(parts)
    
    def _format_approaches_summary(self, papers: List[Dict]) -> str:
        """Summarize different approaches found in papers."""
        approaches = {}
        for paper in papers[:15]:  # Top 15 papers
            for technique in paper['insights'].techniques_used:
                if technique.value not in approaches:
                    approaches[technique.value] = {
                        'count': 0,
                        'avg_quality': 0,
                        'complexities': []
                    }
                approaches[technique.value]['count'] += 1
                approaches[technique.value]['avg_quality'] += paper['insights'].get_quality_score()
                approaches[technique.value]['complexities'].append(
                    paper['insights'].implementation_complexity.value
                )
        
        # Format summary
        summary = "Approaches identified:\n"
        for approach, data in approaches.items():
            avg_quality = data['avg_quality'] / data['count'] if data['count'] > 0 else 0
            summary += f"- {approach}: {data['count']} papers, avg quality {avg_quality:.2f}\n"
        
        return summary
    
    def _extract_year(self, paper_data: Dict) -> int:
        """Extract publication year from paper data."""
        if not paper_data or not paper_data.get('published'):
            return 2020
        try:
            return int(paper_data['published'][:4])
        except:
            return 2020
    
    def _create_fallback_synthesis(self, papers: List[Dict], user_context: UserContext) -> Dict:
        """Create basic synthesis if LLM fails."""
        top_approaches = {}
        for paper in papers[:10]:
            for technique in paper['insights'].techniques_used:
                if technique.value not in top_approaches:
                    top_approaches[technique.value] = {
                        'count': 0,
                        'total_score': 0
                    }
                top_approaches[technique.value]['count'] += 1
                top_approaches[technique.value]['total_score'] += paper.get('rerank_score', 0)
        
        # Sort by score
        sorted_approaches = sorted(
            top_approaches.items(),
            key=lambda x: x[1]['total_score'],
            reverse=True
        )
        
        fallback_text = f"""## Executive Summary

Based on analysis of {len(papers)} research papers, I recommend exploring the following approaches:

## Top Approaches

"""
        for approach, data in sorted_approaches[:3]:
            avg_score = data['total_score'] / data['count'] if data['count'] > 0 else 0
            fallback_text += f"**{approach.replace('_', ' ').title()}**\n"
            fallback_text += f"- Found in {data['count']} high-quality papers\n"
            fallback_text += f"- Average quality score: {avg_score:.2f}\n\n"
        
        fallback_text += """## Next Steps

To proceed with a more detailed analysis, please ensure the LLM service is available. In the meantime, review the research papers directly for specific implementation details."""
        
        return {
            'consultant_analysis': fallback_text,
            'recommendations': {
                'full_text': fallback_text,
                'interactive_options': []
            },
            'metadata': {
                'papers_analyzed': len(papers),
                'synthesis_timestamp': datetime.utcnow().isoformat(),
                'fallback_mode': True
            }
        }
    
    def _create_empty_recommendations(self) -> Dict:
        """Create empty recommendations when no relevant papers found."""
        return {
            'consultant_analysis': """## No Relevant Research Found

Unfortunately, I couldn't find research papers that match your specific context and requirements. This could be because:

1. Your use case is highly specialized or novel
2. The search criteria may be too restrictive
3. This particular application area hasn't been well-researched yet

## Recommendations

1. **Broaden Search Criteria**: Try relaxing some constraints or using more general terms
2. **Consult Industry Experts**: For novel applications, direct consultation may be more valuable
3. **Pilot Approach**: Consider starting with a small proof-of-concept using general best practices

Would you like to adjust your search criteria or explore adjacent research areas?""",
            'recommendations': {
                'full_text': "No relevant papers found for synthesis",
                'interactive_options': [
                    "Broaden search criteria",
                    "Explore adjacent research areas",
                    "Get general GenAI implementation guidance"
                ]
            },
            'metadata': {
                'papers_analyzed': 0,
                'synthesis_timestamp': datetime.utcnow().isoformat()
            }
        }