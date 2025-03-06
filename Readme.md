# AI Competitor Intelligence Agent  

An AI-powered competitor analysis tool using Firecrawl, Exa AI, and Agno's AI Agent framework.  

## Features  

- Extracts and analyzes competitor websites  
- Provides structured comparisons and insights  
- Identifies market gaps, pricing strategies, and growth opportunities  

## Installation  

### 1️⃣ Clone the Repository  

```bash
git clone https://github.com/abhaya1310/ai-competitor-analysis.git
cd ai-competitor-analysis
```

### 2️⃣ Install Dependencies  

```bash
pip install -r requirements.txt
```

### 3️⃣ Set Up API Keys  

Remove '.example' from the .streamlit folder and file inside it, and replace api keys inside secrets.toml with your keys. 

```toml
OPENAI_API_KEY = "your-openai-key"
FIRECRAWL_API_KEY = "your-firecrawl-key"
EXA_API_KEY = "your-exa-key"
```

### 4️⃣ Run the Application  

```bash
streamlit run agent.py
```

## 🎯 Usage  

1️⃣ Enter your **company's URL** or a **brief description**  
2️⃣ Click **Analyze Competitors**  
3️⃣ View the **comparison table** and **analysis report**  

## 🛠 Technologies Used  

- **Streamlit** - For UI and interactivity  
- **Exa AI** - To find similar competitor websites  
- **Firecrawl** - To extract competitor data  
- **Agno AI Agents** - For analysis and comparison  
- **OpenAI GPT-4o-mini** - To generate insights  

## 📬 Contact  
 
- LinkedIn: https://www.linkedin.com/in/abhaya-shukla-1b563521b/

---

