import os
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_CONFIG = {
    "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
    "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
    "deployment_name": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
}

# Pre-defined Career Paths
CAREER_PATHS = {
    "STEM": {
        "subcategories": [
            "Software Engineering", "Data Science", "Artificial Intelligence/Machine Learning",
            "Cybersecurity", "Biotechnology", "Robotics", "Environmental Science",
            "Chemical Engineering", "Electrical Engineering", "Medical Research",
            "Aerospace Engineering", "Mathematics", "Physics", "Computer Science"
        ],
        "description": "Science, Technology, Engineering, and Mathematics fields focusing on analytical thinking, problem-solving, and innovation.",
        "skills": ["Problem-solving", "Analytical thinking", "Technical skills", "Research abilities", "Mathematical reasoning"]
    },
    "Arts": {
        "subcategories": [
            "Digital Art & Design", "Music Production", "Creative Writing", "Film & Video Production",
            "Theater & Performing Arts", "Photography", "Fashion Design", "Architecture",
            "Interior Design", "Game Design", "Animation", "Graphic Design", "Fine Arts"
        ],
        "description": "Creative fields emphasizing artistic expression, creativity, and aesthetic sensibilities.",
        "skills": ["Creativity", "Artistic vision", "Communication", "Aesthetic sense", "Innovation"]
    },
    "Sports": {
        "subcategories": [
            "Professional Athletics", "Sports Management", "Sports Medicine", "Fitness Training",
            "Sports Psychology", "Sports Journalism", "Event Management", "Sports Marketing",
            "Coaching", "Physical Therapy", "Sports Analytics", "Recreational Therapy"
        ],
        "description": "Physical activity and sports-related careers focusing on health, competition, and athletic performance.",
        "skills": ["Physical fitness", "Teamwork", "Leadership", "Discipline", "Competitive spirit"]
    }
}

# Prompt Templates
PROMPT_TEMPLATES = {
    "preference_extraction": """
You are a career guidance counselor. Your task is to extract preferences and interests from a conversation with a student.

Conversation: {conversation}

Please analyze the conversation and extract:
1. Interests and hobbies mentioned
2. Academic subjects they enjoy or excel in
3. Skills they possess or want to develop
4. Work environment preferences (team vs individual, indoor vs outdoor, etc.)
5. Values and motivations
6. Any career-related concerns or goals mentioned

Format your response as a structured summary with clear categories.
""",

    "career_mapping": """
You are an expert career counselor. Based on the extracted preferences, map the student's interests to the most suitable career paths.

Student Preferences: {preferences}

Available Career Categories:
- STEM: {stem_careers}
- Arts: {arts_careers}  
- Sports: {sports_careers}

Please:
1. Rank the top 3 most suitable career categories for this student (STEM, Arts, Sports)
2. For each category, provide a confidence score (0-100%)
3. List specific career subcategories that align with their interests
4. Explain your reasoning for each recommendation

Format as JSON with the following structure:
{{
    "recommendations": [
        {{
            "category": "category_name",
            "confidence_score": 85,
            "specific_careers": ["career1", "career2"],
            "reasoning": "explanation"
        }}
    ]
}}
""",

    "career_explanation": """
You are a career advisor explaining career paths to a student. 

Career Category: {category}
Specific Career: {specific_career}
Student's Interests: {student_interests}

Provide a compelling explanation (2-3 paragraphs) about this career path that includes:
1. What this career involves day-to-day
2. Why it's a good fit for their interests and skills
3. Growth opportunities and potential career progression
4. Required education/skills to get started
5. Potential challenges and rewards

Make it inspiring and personalized to their interests.
""",

    "clarifying_questions": """
Based on the conversation so far, the student's interests seem unclear or too broad for specific career recommendations.

Current conversation: {conversation}
Areas needing clarification: {unclear_areas}

Generate 3-5 strategic questions that will help clarify their interests and preferences. 
Questions should be:
1. Open-ended to encourage detailed responses
2. Focused on uncovering specific interests, values, or skills
3. Designed to differentiate between STEM, Arts, and Sports inclinations
4. Engaging and conversational in tone

Format as a numbered list of questions.
"""
}

# Fallback Questions for Different Scenarios
FALLBACK_QUESTIONS = {
    "general": [
        "What activities make you lose track of time because you enjoy them so much?",
        "When you were a child, what did you dream of becoming when you grew up?",
        "What subjects in school do you find most engaging and why?",
        "Do you prefer working with your hands, your mind, or both?",
        "Are you more energized by working alone or with others?"
    ],
    "stem_exploration": [
        "Do you enjoy solving puzzles or mathematical problems?",
        "Are you curious about how things work or how to build/create new things?",
        "Do you like analyzing data or finding patterns in information?",
        "Are you interested in technology and its impact on society?"
    ],
    "arts_exploration": [
        "Do you enjoy expressing yourself through creative mediums?",
        "Are you drawn to visual arts, performing arts, or written expression?",
        "Do you like telling stories or creating experiences for others?",
        "Are you interested in design, aesthetics, or cultural expression?"
    ],
    "sports_exploration": [
        "Do you enjoy physical activities and staying active?",
        "Are you competitive by nature or do you enjoy team collaboration?",
        "Are you interested in health, fitness, and human performance?",
        "Do you like motivating or coaching others in physical activities?"
    ]
} 