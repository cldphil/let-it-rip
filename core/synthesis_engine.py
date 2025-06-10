"""
Synthesis engine for creating personalized recommendations from paper insights.
Uses a consultant-style LLM approach to match user contexts to relevant research.
"""

from typing import Dict, List, Optional
from datetime import datetime
import logging

from anthropic import Anthropic

from .insight_storage import InsightStorage
from .insight_schema import PaperInsights, UserContext, StudyType

from config import Config

logger = logging.getLogger(__name__)


class SynthesisEngine:
    """
    Creates personalized recommendations using a consultant-style LLM approach.
    
    Emulates McKinsey/Deloitte-style consulting to:
    - Understand client business context
    - Analyze research corpus with special attention to case studies
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
        
        # Step 2: Rerank papers using enhanced reputation metrics
        reranked_papers = self._rerank_papers(relevant_papers)
        
        # Step 3: Select top papers for synthesis (cap at 25 for context window)
        top_papers = reranked_papers[:25]
        
        # Log case study distribution
        case_studies = [p for p in top_papers if p['insights'].study_type == StudyType.CASE_STUDY]
        logger.info(f"Including {len(case_studies)} case studies in synthesis")
        
        # Step 4: Generate consultant-style synthesis
        synthesis_result = self._generate_consultant_synthesis(
            top_papers, 
            user_context,
            interactive
        )
        
        return synthesis_result
    
    def _rerank_papers(self, papers: List[Dict]) -> List[Dict]:
        """
        New formula: 
        final_score = 0.35 * reputation + 0.25 * recency + 0.20 * case_study_bonus + 0.20 * validation_bonus
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
            
            # Case study bonus - significant weight for real-world implementations
            case_study_score = 0.0
            if insights.study_type == StudyType.CASE_STUDY:
                case_study_score = 1
            
            # Reputation score from insights (already includes author h-index and conference validation)
            reputation_score = insights.get_reputation_score()
            
            # Calculate final reranking score with updated algorithm
            paper_data['rerank_score'] = (
                0.35 * reputation_score +          # Reputation (includes author h-index, conference)
                0.25 * recency_score +          # Recency (recent research prioritized)
                0.20 * case_study_score       # Case study implementation bonus
            )
            
            # Store components for transparency
            paper_data['score_components'] = {
                'reputation': reputation_score,
                'recency': recency_score,
                'case_study': case_study_score
            }
        
        # Sort by rerank score
        papers.sort(key=lambda x: x['rerank_score'], reverse=True)
        return papers
    
    def _generate_consultant_synthesis(self, 
                                     papers: List[Dict], 
                                     user_context: UserContext,
                                     interactive: bool = False) -> Dict:
        """
        Generate McKinsey-style synthesis using LLM with case study awareness.
        """
        # Prepare research summaries with case study highlighting
        research_summaries = self._prepare_research_summaries(papers)
        
        # Separate case studies for special attention
        case_studies = [s for s in research_summaries if s.get('is_case_study')]
        other_studies = [s for s in research_summaries if not s.get('is_case_study')]
        
        # Create consultant prompt
        prompt = self._create_consultant_prompt(
            user_context, 
            case_studies, 
            other_studies, 
            interactive
        )
        
        try:
            response = self.llm.messages.create(
                model=Config.LLM_MODEL,
                temperature=0.3,  # Lower temperature for professional consistency
                max_tokens=3000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse the structured response
            synthesis = self._parse_consultant_response(response.content[0].text)
            
            # Add enhanced metadata
            synthesis['metadata'] = {
                'papers_analyzed': len(papers),
                'case_studies_included': len(case_studies),
                'synthesis_timestamp': datetime.utcnow().isoformat(),
                'top_paper_scores': [
                    {
                        'title': self.storage.load_paper(p['paper_id']).get('title', 'Unknown')[:100],
                        'score': p['rerank_score'],
                        'components': p['score_components'],
                        'is_case_study': p['insights'].study_type == StudyType.CASE_STUDY
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
        Prepare research summaries with enhanced handling of rich key findings.
        Updated to remove references to deprecated fields.
        """
        summaries = []
        
        for i, paper_data in enumerate(papers):
            insights = paper_data['insights']
            paper_info = self.storage.load_paper(paper_data['paper_id'])
            
            if not paper_info:
                continue
            
            # Create enhanced findings summary
            key_findings_text = self._create_enhanced_findings_summary(insights)
            
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
                'reputation_score': insights.get_reputation_score(),
                'complexity': insights.implementation_complexity.value,
                'published_year': self._extract_year(paper_info),
                'is_case_study': insights.study_type == StudyType.CASE_STUDY,
                'has_code': insights.has_code_available,
                'key_findings_count': len(insights.key_findings),
                'study_type': insights.study_type.value
            }
            
            summaries.append(summary)
        
        return summaries
    
    def _create_enhanced_findings_summary(self, insights: PaperInsights) -> str:
        """
        Create a concise summary leveraging richer key findings from enhanced extraction.
        """
        findings_text = ""
        
        # Since findings are now 1-3 sentences each, prioritize by content
        quantitative_findings = []
        implementation_findings = []
        outcome_findings = []
        other_findings = []
        
        for finding in insights.key_findings:
            finding_lower = finding.lower()
            # Categorize findings
            if any(char.isdigit() for char in finding) and '%' in finding:
                quantitative_findings.append(finding)
            elif any(term in finding_lower for term in ['implement', 'deploy', 'system', 'architecture']):
                implementation_findings.append(finding)
            elif any(term in finding_lower for term in ['result', 'outcome', 'improve', 'reduce']):
                outcome_findings.append(finding)
            else:
                other_findings.append(finding)
        
        # Build summary with best findings from each category
        selected_findings = []
        if quantitative_findings:
            selected_findings.append(quantitative_findings[0])
        if outcome_findings:
            selected_findings.append(outcome_findings[0])
        if implementation_findings:
            selected_findings.append(implementation_findings[0])
        
        # Fill remaining space with other findings
        remaining_slots = 3 - len(selected_findings)
        if remaining_slots > 0 and other_findings:
            selected_findings.extend(other_findings[:remaining_slots])
        
        findings_text = " ".join(selected_findings)
        
        # Add context tags
        prefix = ""
        if insights.study_type == StudyType.CASE_STUDY:
            prefix = "[CASE STUDY] "
        
        findings_text = prefix + findings_text
        
        # Add problem context if short
        if len(findings_text) < 500 and insights.problem_addressed:
            findings_text = f"Addresses: {insights.problem_addressed}. {findings_text}"
        
        # Truncate to ~200 words (roughly 1000 characters)
        if len(findings_text) > 1000:
            findings_text = findings_text[:997] + "..."
        
        return findings_text
    
    def _create_consultant_prompt(self, 
                                user_context: UserContext, 
                                case_studies: List[Dict],
                                other_studies: List[Dict],
                                interactive: bool) -> str:
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
        
        # Format case studies separately
        case_studies_text = ""
        if case_studies:
            case_studies_text = "\n### Real-World Case Studies (Highest Priority)\n"
            for summary in case_studies[:5]:  # Top 5 case studies
                validation_tag = "Case Study"
                case_studies_text += f"""
                    {summary['rank']}. [{validation_tag}] {summary['title']}
                    Approach: {summary['approach']}
                    Key Findings: {summary['findings']}
                    Limitations: {summary['limitations']}
                    Reputation Score: {summary['reputation_score']:.2f} | Complexity: {summary['complexity']} | Year: {summary['published_year']}
                    Key Findings Count: {summary['key_findings_count']} | Study Type: {summary['study_type']}
                    """

        # Format other research
        other_studies_text = ""
        if other_studies:
            other_studies_text = "\n### Additional Research Studies\n"
            for summary in other_studies[:10]:  # Top 10 other studies
                code_tag = "[Code Available] " if summary.get('has_code') else ""
                other_studies_text += f"""
{summary['rank']}. {code_tag}{summary['title']}
   Approach: {summary['approach']}
   Key Findings: {summary['findings']}
   Reputation Score: {summary['reputation_score']:.2f} | Complexity: {summary['complexity']} | Year: {summary['published_year']}
   Study Type: {summary['study_type']} | Key Findings: {summary['key_findings_count']}
"""

        # Create consultant prompt with updated metrics
        prompt = f"""You are a senior consultant at McKinsey & Company specializing in AI strategy and implementation. Your client has approached you for guidance on implementing generative AI solutions.

## Client Context
{context_text}

## Research Analysis
I've analyzed {len(case_studies) + len(other_studies)} relevant research papers, including {len(case_studies)} real-world case studies.

{case_studies_text}
{other_studies_text}

## Your Consulting Tasks

Please provide a strategic recommendation following this structure:

1. **Executive Summary** (2-3 sentences)
   - What is your primary recommendation?
   - Why is this the best fit for the client?
   - Highlight if recommendation is based on proven case studies

2. **Recommended Approach**
   - Primary technique/methodology recommended
   - Why this approach aligns with client constraints
   - Expected complexity and resource requirements
   - Reference specific case studies if applicable

3. **Evidence from Case Studies** (if applicable)
   - What real-world implementations support this recommendation?
   - What were their outcomes?
   - How do they relate to the client's context?

4. **Strategic Advantages**
   - 3-4 key benefits specific to the client's context
   - Competitive advantages this approach provides
   - ROI indicators based on case study evidence

5. **Risk Assessment**
   - 2-3 primary risks or challenges
   - Mitigation strategies for each risk
   - Lessons learned from case studies

6. **Implementation Considerations**
   - High-level phases (do not detail full roadmap yet)
   - Key prerequisites or dependencies
   - Recommended team composition
   - Timeline expectations based on similar implementations

{"7. **Next Steps**" if interactive else "7. **Getting Started**"}
{" - Would you like me to develop a detailed implementation roadmap?" if interactive else " - Key actions for the first 30 days"}
{" - Are there alternative approaches you'd like to explore?" if interactive else " - Quick wins to build momentum"}
{" - Do you have specific concerns about this recommendation?" if interactive else " - Success metrics to track"}

IMPORTANT: 
- Give strong preference to recommendations backed by real-world case studies, especially industry-validated ones
- Focus on reputation scores (which include author credibility and conference validation) rather than subjective metrics
- Consider the number and depth of key findings as indicators of research thoroughness
- If no relevant case studies exist, note this as a consideration and recommend pilot approaches

Please structure your response in a clear, professional format using markdown. Be specific and actionable, referencing the research evidence where appropriate."""

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
                    "Deep dive on specific case studies",
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
        Explore alternative approaches with case study awareness.
        """
        # Categorize papers by approach and case study status
        approaches_data = self._categorize_approaches(papers)
        
        prompt = f"""As the McKinsey consultant, the client has asked to explore alternative approaches beyond the primary recommendation.

## Client Context
{self._format_user_context(user_context)}

## Available Approaches from Research
{self._format_categorized_approaches(approaches_data)}

Please provide {num_alternatives} alternative approaches, each optimized for different priorities:

For each alternative, provide:
1. **Approach Name & Rationale**
   - What priority does this optimize for? (e.g., speed, cost, risk, innovation)
   - Key differentiator from primary recommendation
   - Case study evidence (if available)

2. **Pros and Cons**
   - Specific advantages for this client
   - Trade-offs compared to primary recommendation
   - Real-world validation status

3. **Best Fit Scenario**
   - When to choose this approach
   - Conditions for success
   - Similar implementations

4. **Implementation Complexity**
   - Relative to primary recommendation
   - Key challenges to expect
   - Resource requirements

Prioritize alternatives that have case study support where possible.

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
    
    def _categorize_approaches(self, papers: List[Dict]) -> Dict:
        """Categorize approaches with case study information."""
        approaches = {}
        
        for paper in papers[:20]:  # Top 20 papers
            insights = paper['insights']
            for technique in insights.techniques_used:
                if technique.value not in approaches:
                    approaches[technique.value] = {
                        'count': 0,
                        'case_studies': 0,
                        'avg_reputation': 0,
                        'complexities': [],
                        'validated_cases': []
                    }
                
                approaches[technique.value]['count'] += 1
                approaches[technique.value]['avg_reputation'] += insights.get_reputation_score()
                approaches[technique.value]['complexities'].append(
                    insights.implementation_complexity.value
                )
                
                if insights.study_type == StudyType.CASE_STUDY:
                    approaches[technique.value]['case_studies'] += 1
        
        # Calculate averages
        for approach_data in approaches.values():
            if approach_data['count'] > 0:
                approach_data['avg_reputation'] /= approach_data['count']
        
        return approaches
    
    def _format_categorized_approaches(self, approaches_data: Dict) -> str:
        """Format categorized approaches with case study information."""
        summary = "Approaches identified with case study evidence:\n\n"
        
        # Sort by number of case studies, then by count
        sorted_approaches = sorted(
            approaches_data.items(),
            key=lambda x: (x[1]['case_studies'], x[1]['count']),
            reverse=True
        )
        
        for approach, data in sorted_approaches:
            approach_name = approach.replace('_', ' ').title()
            summary += f"**{approach_name}**\n"
            summary += f"- Papers: {data['count']} (including {data['case_studies']} case studies)\n"
            summary += f"- Avg Reputation Score: {data['avg_reputation']:.2f}\n"
            
            if data['validated_cases']:
                summary += f"- Industry Validated Cases:\n"
                for case in data['validated_cases'][:2]:  # Show top 2
                    summary += f"  • {case}\n"
            
            summary += "\n"
        
        return summary
    
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
    
    def _extract_year(self, paper_data: Dict) -> int:
        """Extract publication year from paper data."""
        if not paper_data or not paper_data.get('published'):
            return 2020
        try:
            return int(paper_data['published'][:4])
        except:
            return 2020
    
    def _create_fallback_synthesis(self, papers: List[Dict], user_context: UserContext) -> Dict:
        """Create enhanced fallback synthesis with case study awareness."""
        # Separate case studies
        case_studies = [p for p in papers if p['insights'].study_type == StudyType.CASE_STUDY]
        other_papers = [p for p in papers if p['insights'].study_type != StudyType.CASE_STUDY]
        
        top_approaches = {}
        for paper in papers[:10]:
            for technique in paper['insights'].techniques_used:
                if technique.value not in top_approaches:
                    top_approaches[technique.value] = {
                        'count': 0,
                        'total_score': 0,
                        'case_study_count': 0
                    }
                top_approaches[technique.value]['count'] += 1
                top_approaches[technique.value]['total_score'] += paper.get('rerank_score', 0)
                if paper['insights'].study_type == StudyType.CASE_STUDY:
                    top_approaches[technique.value]['case_study_count'] += 1
        
        # Sort by score
        sorted_approaches = sorted(
            top_approaches.items(),
            key=lambda x: x[1]['total_score'],
            reverse=True
        )
        
        fallback_text = f"""## Executive Summary

Based on analysis of {len(papers)} research papers (including {len(case_studies)} real-world case studies), I recommend exploring the following approaches:

## Top Approaches

"""
        for approach, data in sorted_approaches[:3]:
            avg_score = data['total_score'] / data['count'] if data['count'] > 0 else 0
            approach_name = approach.replace('_', ' ').title()
            fallback_text += f"**{approach_name}**\n"
            fallback_text += f"- Found in {data['count']} high-reputation papers"
            if data['case_study_count'] > 0:
                fallback_text += f" (including {data['case_study_count']} case studies)"
            fallback_text += f"\n- Average reputation score: {avg_score:.2f}\n\n"
        
        if case_studies:
            fallback_text += f"""## Notable Case Studies

The following real-world implementations provide valuable insights:

"""
            for cs in case_studies[:3]:
                paper_info = self.storage.load_paper(cs['paper_id'])
                if paper_info:
                    fallback_text += f"- {paper_info.get('title', 'Unknown')[:100]}\n"
        
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
                'case_studies_included': len(case_studies),
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
                'case_studies_included': 0,
                'synthesis_timestamp': datetime.utcnow().isoformat()
            }
        }