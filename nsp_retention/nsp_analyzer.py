import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import base64
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel
from langchain_groq import ChatGroq


class NSPAnalyzer:
    """NSP Analyzer class for FastAPI implementation"""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with DataFrame"""
        # Fix the Phone number column issue by converting to string
        if 'Phone number' in df.columns:
            df['Phone number'] = df['Phone number'].astype(str)
        self.df = df
        self.subject_outcomes = None
    
    def standardize_programs(self):
        """Standardize program names for consistency"""
        # Map similar programs to standardized names
        program_mapping = {
            'Computer Science': 'Computer Science',
            'Information Technology': 'Information Technology',
            'Computing With Accounting': 'Computing With Accounting',
            'Computer Engineering': 'Computer Engineering',
            'Information and Communication Technology': 'Information Technology',
            
        }
        
        # Apply mapping where exact matches exist
        self.df['Standardized Program'] = self.df['Program'].apply(
            lambda x: next((v for k, v in program_mapping.items() if k.lower() in x.lower()), x)
            if isinstance(x, str) else "Unknown"
        )
    
    def analyze_hiring_success(self):
        """Analyze hiring success rates by subject specialization"""
        # Clean up data
        self.df['Program'] = self.df['Program'].fillna('Unknown')
        self.df['Current status'] = self.df['Current status'].fillna('Unknown')
        
        # Standardize programs
        self.standardize_programs()
        
        # Calculate metrics per program
        success_metrics = []
        
        # Group by subject specialization
        grouped = self.df.groupby('Standardized Program')
        
        for program, group in grouped:
            total_candidates = len(group)
            hired = len(group[group['Current status'] == 'Hired'])
            not_hired = len(group[group['Current status'] == 'Not Hired'])
            offered_bootcamp = len(group[group['Current status'] == 'Offered Bootcamp'])
            
            # Calculate success rates
            hire_rate = (hired / total_candidates * 100) if total_candidates > 0 else 0
            
            success_metrics.append({
                'Subject': program,
                'Total Candidates': total_candidates,
                'Hired': hired,
                'Not Hired': not_hired,
                'Offered Bootcamp': offered_bootcamp,
                'Hire Rate (%)': round(hire_rate, 2)
            })
        
        # Create DataFrame from metrics
        self.subject_outcomes = pd.DataFrame(success_metrics)
        
        # Sort by hire rate
        self.subject_outcomes = self.subject_outcomes.sort_values('Hire Rate (%)', ascending=False)
        
        return self.subject_outcomes

    def get_overall_stats(self) -> Dict[str, Any]:
        """Calculate and return overall statistics"""
        if self.subject_outcomes is None or self.subject_outcomes.empty:
            return {
                "top_subject": "N/A",
                "avg_rate": 0.0,
                "total_candidates": 0,
             
            }
        
        top_subject = self.subject_outcomes.iloc[0]['Subject']
        top_rate = self.subject_outcomes.iloc[0]['Hire Rate (%)']
        avg_rate = self.subject_outcomes['Hire Rate (%)'].mean()
        total_candidates = self.subject_outcomes['Total Candidates'].sum()
        advantage = top_rate - avg_rate
        
        
        return {
            "top_subject": top_subject,
            "avg_rate": round(avg_rate, 2),
            "total_candidates": total_candidates,
          
        }


class NSPVisualizer:
    """NSP Visualizer for FastAPI implementation"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def visualize_subject_success_rates(self) -> str:
        """Create a bar chart of hiring success rates by subject and return as base64 string"""
        # Group by program and calculate success rates
        program_stats = []
        for program, group in self.df.groupby('Standardized Program'):
            if len(group) < 2:  # Skip programs with too few candidates
                continue
                
            total = len(group)
            hired = len(group[group['Current status'] == 'Hired'])
            hire_rate = (hired / total * 100) if total > 0 else 0
            
            program_stats.append({
                'Subject': program,
                'Hire Rate (%)': hire_rate,
                'Total Candidates': total
            })
        
        # Convert to DataFrame
        stats_df = pd.DataFrame(program_stats)
        if stats_df.empty:
            return ""
            
        stats_df = stats_df.sort_values('Hire Rate (%)', ascending=False)
        
        # Create the visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the bars with proper hue parameter
        bars = sns.barplot(
            x='Subject', 
            y='Hire Rate (%)',
            hue='Subject',
            data=stats_df,
            palette='viridis',
            legend=False,
            ax=ax
        )
        
        # Add average line
        avg_rate = stats_df['Hire Rate (%)'].mean()
        ax.axhline(avg_rate, color='red', linestyle='--', alpha=0.7, 
                   label=f'Average: {avg_rate:.1f}%')
        
        # Add value labels on top of bars
        for i, bar in enumerate(bars.patches):
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                bar.get_height() + 1,
                f"{bar.get_height():.1f}%",
                ha='center', va='bottom', color='black', fontweight='bold'
            )
        
        # Customize the chart
        ax.set_title('NSP Hiring Success Rate by Subject Specialization', fontsize=16)
        ax.set_xlabel('Subject', fontsize=14)
        ax.set_ylabel('Hire Rate (%)', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        
        # Add legend
        ax.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert to base64 string
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        
        # Encode as base64
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return img_str
    
    def visualize_retention_comparison(self) -> str:
        """Create a visualization comparing retention by subject and return as base64 string"""
        # Group data by program
        program_data = {}
        
        for program, group in self.df.groupby('Standardized Program'):
            if len(group) < 2:  # Skip programs with too few candidates
                continue
                
            # Count each outcome
            outcomes = group['Current status'].value_counts().to_dict()
            total = len(group)
            
            # Calculate percentages
            hired_pct = (outcomes.get('Hired', 0) / total * 100) if total > 0 else 0
            bootcamp_pct = (outcomes.get('Offered Bootcamp', 0) / total * 100) if total > 0 else 0
            not_hired_pct = (outcomes.get('Not Hired', 0) / total * 100) if total > 0 else 0
            
            program_data[program] = {
                'Hired': hired_pct,
                'Offered Bootcamp': bootcamp_pct,
                'Not Hired': not_hired_pct,
                'Total': total
            }
        
        # Convert to DataFrame
        comparison_df = pd.DataFrame(program_data).T
        if comparison_df.empty:
            return ""
            
        comparison_df = comparison_df.sort_values('Hired', ascending=False)
        
        # Create the visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create stacked bar chart
        comparison_df[['Hired', 'Offered Bootcamp', 'Not Hired']].plot(
            kind='bar', 
            stacked=True,
            colormap='viridis',
            ax=ax
        )
        
        # Customize the chart
        ax.set_title('NSP Outcomes by Subject Specialization', fontsize=16)
        ax.set_xlabel('Subject', fontsize=14)
        ax.set_ylabel('Percentage (%)', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Outcome')
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert to base64 string
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        
        # Encode as base64
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return img_str


def generate_recommendations(subject_data: pd.DataFrame, api_key: str, top_n: int = 3) -> List[str]:
    """Generate recommendations using LangChain and Groq"""
    # Handle empty data
    if subject_data.empty or len(subject_data) < top_n:
        return ["Not enough data to generate recommendations."]
    
    # Get top subjects
    top_subjects = subject_data.nlargest(top_n, 'Hire Rate (%)')
    
    # Prepare prompt
    prompt = (
        "You are an HR analytics expert reviewing National Service Personnel (NSP) "
        "hiring data. The following table shows hiring success rates by subject specialization:\n\n"
        f"{top_subjects.to_string()}\n\n"
        "Based on this data, provide 3 specific, actionable recommendations for HR to improve "
        "hiring success. Focus on which subjects are most aligned with long-term success.\n\n"
        "Format each recommendation on a new line, highlighting the specific percentage advantage "
        "of the top subjects (e.g., 'NSPs with degrees in Subject Y are 20% more likely to be retained')."
    )
    
    try:
        # Initialize LLM
        llm = ChatGroq(
            model="llama-3.1-8b-instant", 
            api_key=api_key
        )
        
        # Use the chain-style invocation
        response = llm.invoke(prompt)
        
        # Get the response content
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Split into individual recommendations
        recommendations = [rec.strip() for rec in response_text.split('\n') if rec.strip()]
        
        # If no recommendations were generated, provide a fallback
        if not recommendations:
            return ["Could not generate specific recommendations. Please check your data and try again."]
            
        return recommendations
    except Exception as e:
        return [f"Error generating recommendations: {str(e)}"]


def generate_report(subject_outcomes: pd.DataFrame, recommendations: List[str]) -> str:
    """Generate a markdown report"""
    if subject_outcomes.empty:
        return "Not enough data to generate a report."
    
    # Find top subject and its advantage
    top_subject = subject_outcomes.iloc[0]['Subject']
    top_rate = subject_outcomes.iloc[0]['Hire Rate (%)']
    avg_rate = subject_outcomes['Hire Rate (%)'].mean()
    advantage = top_rate - avg_rate
    
    report_md = f"""# NSP Hiring Success & Retention Analysis Report

## Executive Summary

This analysis examines the correlation between subject specialization and post-term employment rates for National Service Personnel (NSP). The data reveals significant variations in hiring success rates across different academic backgrounds, providing actionable insights for HR recruitment strategies.

## Key Findings

- NSPs with degrees in **{top_subject}** are **{advantage:.1f}%** more likely to be retained than average.
- The average hiring success rate across all specializations is **{avg_rate:.1f}%**.
- Significant variation exists between the highest performing subject ({top_rate:.1f}%) and lowest performing subject ({subject_outcomes.iloc[-1]['Hire Rate (%)']:.1f}%).

## Subject Specialization Success Metrics

{subject_outcomes.to_markdown(index=False)}

"""
    
    if recommendations:
        report_md += "\n## HR Recommendations\n\n"
        for i, rec in enumerate(recommendations, 1):
            report_md += f"{i}. {rec}\n\n"
    
    report_md += "\n\n---\n*Report generated using LangChain and Groq*\n"
    
    return report_md