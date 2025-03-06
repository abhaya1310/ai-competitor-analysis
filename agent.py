import streamlit as st
from exa_py import Exa
from agno.agent import Agent
from agno.tools.firecrawl import FirecrawlTools
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
import pandas as pd
import requests
from firecrawl import FirecrawlApp
from pydantic import BaseModel, Field
from typing import List, Optional
import json

# Streamlit UI
st.set_page_config(page_title="AI Competitor Intelligence Agent Team", layout="wide")

# Load API keys from secrets.toml in .streamlit
openai_api_key = st.secrets["OPENAI_API_KEY"]
firecrawl_api_key = st.secrets["FIRECRAWL_API_KEY"]
exa_api_key = st.secrets["EXA_API_KEY"]

st.title(" AI Competitor Analysis Agent ")
st.info(
    """
        This app uses AI to analyze competitor websites. Just provide your company's URL and/or a brief description, and you'll receive a detailed analysis report.\n
        For best results, include both your company URL and description.
    """
)


# Input fields for URL and description
url = st.text_input("Enter your company URL :")
description = st.text_area("Enter a description of your company (if URL is not available):")

# Initialize Exa AI and Firecrawl tools using API keys from secrets
exa = Exa(api_key=exa_api_key)

firecrawl_tools = FirecrawlTools(
    api_key=firecrawl_api_key,
    scrape=False,
    crawl=True,
    limit=5
)

firecrawl_agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini", api_key=openai_api_key),
    tools=[firecrawl_tools, DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True
)

analysis_agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini", api_key=openai_api_key),
    show_tool_calls=True,
    markdown=True
)

# New agent for comparing competitor data
comparison_agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini", api_key=openai_api_key),
    show_tool_calls=True,
    markdown=True
)

def get_competitor_urls(url: str = None, description: str = None) -> list[str]:
    if not url and not description:
        raise ValueError("Please provide either a URL or a description.")

    # Only Exa AI is used now.
    try:
        if url:
            result = exa.find_similar(
                url=url,
                num_results=3,
                exclude_source_domain=True,
                category="company"
            )
        else:
            result = exa.search(
                description,
                type="neural",
                category="company",
                use_autoprompt=True,
                num_results=3
            )
        return [item.url for item in result.results]
    except Exception as e:
        st.error(f"Error fetching competitor URLs from Exa: {str(e)}")
        return []

class CompetitorDataSchema(BaseModel):
    company_name: str = Field(description="Name of the company")
    pricing: str = Field(description="Pricing details, tiers, and plans")
    key_features: List[str] = Field(description="Main features and capabilities of the product/service")
    tech_stack: List[str] = Field(description="Technologies, frameworks, and tools used")
    marketing_focus: str = Field(description="Main marketing angles and target audience")
    customer_feedback: str = Field(description="Customer testimonials, reviews, and feedback")

def extract_competitor_info(competitor_url: str) -> Optional[dict]:
    try:
        # Initialize FirecrawlApp with API key
        app = FirecrawlApp(api_key=firecrawl_api_key)
        
        # Add wildcard to crawl subpages
        url_pattern = f"{competitor_url}/*"
        
        extraction_prompt = """
        Extract detailed information about the company's offerings, including:
        - Company name and basic information
        - Pricing details, plans, and tiers
        - Key features and main capabilities
        - Technology stack and technical details
        - Marketing focus and target audience
        - Customer feedback and testimonials
        
        Analyze the entire website content to provide comprehensive information for each field.
        """
        
        response = app.extract(
            [url_pattern],
            {
                'prompt': extraction_prompt,
                'schema': CompetitorDataSchema.model_json_schema(),
            }
        )
        
        if response.get('success') and response.get('data'):
            extracted_info = response['data']
            
            # Create JSON structure
            competitor_json = {
                "competitor_url": competitor_url,
                "company_name": extracted_info.get('company_name', 'N/A'),
                "pricing": extracted_info.get('pricing', 'N/A'),
                "key_features": extracted_info.get('key_features', [])[:5],
                "tech_stack": extracted_info.get('tech_stack', [])[:5],
                "marketing_focus": extracted_info.get('marketing_focus', 'N/A'),
                "customer_feedback": extracted_info.get('customer_feedback', 'N/A')
            }
            
            return competitor_json
        else:
            return None
            
    except Exception as e:
        return None

def generate_comparison_report(competitor_data: list) -> None:
    formatted_data = json.dumps(competitor_data, indent=2)
    print(formatted_data)
    
    system_prompt = f"""
    As an expert business analyst, analyze the following competitor data in JSON format and create a structured comparison.
    Extract and summarize the key information into concise points.

    {formatted_data}

    Return the data in a structured format with EXACTLY these columns:
    Company, Pricing, Key Features, Tech Stack, Marketing Focus, Customer Feedback

    Rules:
    1. For Company: Include company name and URL
    2. For Key Features: List top 3 most important features only
    3. For Tech Stack: List top 3 most relevant technologies only
    4. Keep all entries clear and concise
    5. Format feedback as brief quotes
    6. Return ONLY the structured data, no additional text
    """

    # Get comparison data from agent
    comparison_response = comparison_agent.run(system_prompt)
    
    try:
        # Split the response into lines and clean them
        table_lines = [
            line.strip() 
            for line in comparison_response.content.split('\n') 
            if line.strip() and '|' in line
        ]
        
        # Extract headers (first row)
        headers = [
            col.strip() 
            for col in table_lines[0].split('|') 
            if col.strip()
        ]
        
        # Extract data rows (skip header and separator rows)
        data_rows = []
        for line in table_lines[2:]:
            row_data = [
                cell.strip() 
                for cell in line.split('|') 
                if cell.strip()
            ]
            if len(row_data) == len(headers):
                data_rows.append(row_data)
        
        # Create DataFrame
        df = pd.DataFrame(
            data_rows,
            columns=headers
        )
        
        # Display the table
        st.subheader("Competitor Comparison")
        st.table(df)
        
    except Exception as e:
        st.error(f"Error creating comparison table: {str(e)}")
        st.write("Raw comparison data for debugging:", comparison_response.content)

def generate_analysis_report(competitor_data: list):
    formatted_data = json.dumps(competitor_data, indent=2)
    print("Analysis Data:", formatted_data)  # For debugging
    
    report = analysis_agent.run(
        f"""Analyze the following competitor data in JSON format and identify market opportunities to improve my own company:
        
        {formatted_data}

        Tasks:
        1. Identify market gaps and opportunities based on competitor offerings
        2. Analyze competitor weaknesses that we can capitalize on
        3. Recommend unique features or capabilities we should develop
        4. Suggest pricing and positioning strategies to gain competitive advantage
        5. Outline specific growth opportunities in underserved market segments
        6. Provide actionable recommendations for product development and go-to-market strategy

        Focus on finding opportunities where we can differentiate and do better than competitors.
        Highlight any unmet customer needs or pain points we can address.
        """
    )
    return report.content

if st.button("Analyze Competitors"):
    if url or description:
        with st.spinner("Fetching competitor URLs..."):
            competitor_urls = get_competitor_urls(url=url, description=description)
            st.write(f"Competitor URLs: {competitor_urls}")
        
        competitor_data = []
        for comp_url in competitor_urls:
            with st.spinner(f"Analyzing Competitor: {comp_url}..."):
                competitor_info = extract_competitor_info(comp_url)
                if competitor_info is not None:
                    competitor_data.append(competitor_info)
        
        if competitor_data:
            with st.spinner("Generating comparison table..."):
                generate_comparison_report(competitor_data)
            
            with st.spinner("Generating analysis report..."):
                analysis_report = generate_analysis_report(competitor_data)
                st.subheader("Competitor Analysis Report")
                st.markdown(analysis_report)
            
            st.success("Analysis complete!")
        else:
            st.error("Could not extract data from any competitor URLs")
    else:
        st.error("Please provide either a URL or a description.")
