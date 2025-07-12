import json
import openai
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import config

@dataclass
class ConversationAnalysis:
    """Data class to store conversation analysis results"""
    interests: List[str]
    academic_preferences: List[str]
    skills: List[str]
    work_environment: List[str]
    values: List[str]
    career_goals: List[str]
    confidence_score: float

@dataclass
class CareerRecommendation:
    """Data class to store career recommendation results"""
    category: str
    specific_careers: List[str]
    confidence_score: float
    reasoning: str
    match_score: float

class CareerAdvisorAI:
    """Main class for AI-powered career advisory system"""
    
    def __init__(self):
        """Initialize the Career Advisor with Azure OpenAI client and embeddings"""
        self.client = openai.AzureOpenAI(
            api_key=config.AZURE_OPENAI_CONFIG["api_key"],
            api_version=config.AZURE_OPENAI_CONFIG["api_version"],
            azure_endpoint=config.AZURE_OPENAI_CONFIG["azure_endpoint"]
        )
        
        # Initialize sentence transformer for embeddings
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create career embeddings for semantic matching
        self._create_career_embeddings()
        
        # Conversation history
        self.conversation_history = []
    
    def _create_career_embeddings(self):
        """Create embeddings for all career paths for semantic matching"""
        career_texts = []
        career_labels = []
        
        for category, info in config.CAREER_PATHS.items():
            for career in info["subcategories"]:
                career_text = f"{career} {info['description']} {' '.join(info['skills'])}"
                career_texts.append(career_text)
                career_labels.append((category, career))
        
        self.career_embeddings = self.sentence_model.encode(career_texts)
        self.career_labels = career_labels
    
    def _call_azure_openai(self, prompt: str, max_tokens: int = 1000) -> str:
        """Make a call to Azure OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=config.AZURE_OPENAI_CONFIG["deployment_name"],
                messages=[
                    {"role": "system", "content": "You are an expert career counselor with deep knowledge of various career paths."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling Azure OpenAI: {str(e)}"
    
    def extract_preferences(self, conversation: str) -> ConversationAnalysis:
        """Extract preferences and interests from conversation"""
        prompt = config.PROMPT_TEMPLATES["preference_extraction"].format(
            conversation=conversation
        )
        
        response = self._call_azure_openai(prompt)
        
        # Parse the response (simplified version - in production, use more robust parsing)
        # For demo purposes, we'll extract key information
        interests = self._extract_list_from_response(response, "interests")
        academic = self._extract_list_from_response(response, "academic")
        skills = self._extract_list_from_response(response, "skills")
        work_env = self._extract_list_from_response(response, "environment")
        values = self._extract_list_from_response(response, "values")
        goals = self._extract_list_from_response(response, "goals")
        
        return ConversationAnalysis(
            interests=interests,
            academic_preferences=academic,
            skills=skills,
            work_environment=work_env,
            values=values,
            career_goals=goals,
            confidence_score=0.8  # Simplified confidence scoring
        )
    
    def _extract_list_from_response(self, response: str, category: str) -> List[str]:
        """Helper method to extract lists from AI response"""
        lines = response.lower().split('\n')
        items = []
        in_category = False
        
        for line in lines:
            if category in line and ':' in line:
                in_category = True
                # Extract items from the same line if present
                after_colon = line.split(':', 1)[1].strip()
                if after_colon:
                    items.extend([item.strip() for item in after_colon.split(',') if item.strip()])
            elif in_category and line.strip():
                if line.startswith('-') or line.startswith('•'):
                    items.append(line.strip(' -•').strip())
                elif not any(cat in line for cat in ['interests', 'academic', 'skills', 'environment', 'values', 'goals']):
                    items.append(line.strip())
                else:
                    in_category = False
        
        return items[:5]  # Limit to top 5 items
    
    def map_to_career_paths(self, preferences: ConversationAnalysis) -> List[CareerRecommendation]:
        """Map student preferences to career paths using AI and semantic similarity"""
        
        # Prepare career information for the prompt
        stem_careers = ", ".join(config.CAREER_PATHS["STEM"]["subcategories"][:10])
        arts_careers = ", ".join(config.CAREER_PATHS["Arts"]["subcategories"][:10])
        sports_careers = ", ".join(config.CAREER_PATHS["Sports"]["subcategories"][:10])
        
        preferences_text = f"""
        Interests: {', '.join(preferences.interests)}
        Academic Preferences: {', '.join(preferences.academic_preferences)}
        Skills: {', '.join(preferences.skills)}
        Work Environment: {', '.join(preferences.work_environment)}
        Values: {', '.join(preferences.values)}
        Career Goals: {', '.join(preferences.career_goals)}
        """
        
        prompt = config.PROMPT_TEMPLATES["career_mapping"].format(
            preferences=preferences_text,
            stem_careers=stem_careers,
            arts_careers=arts_careers,
            sports_careers=sports_careers
        )
        
        response = self._call_azure_openai(prompt, max_tokens=1500)
        
        # Also perform semantic similarity matching
        semantic_matches = self._get_semantic_matches(preferences_text)
        
        # Parse AI response and combine with semantic matches
        recommendations = self._parse_career_recommendations(response, semantic_matches)
        
        return recommendations
    
    def _get_semantic_matches(self, preferences_text: str) -> List[Tuple[str, str, float]]:
        """Get career matches using semantic similarity"""
        pref_embedding = self.sentence_model.encode([preferences_text])
        similarities = cosine_similarity(pref_embedding, self.career_embeddings)[0]
        
        # Get top 10 matches
        top_indices = np.argsort(similarities)[-10:][::-1]
        matches = []
        
        for idx in top_indices:
            category, career = self.career_labels[idx]
            score = similarities[idx]
            matches.append((category, career, score))
        
        return matches
    
    def _parse_career_recommendations(self, ai_response: str, semantic_matches: List[Tuple[str, str, float]]) -> List[CareerRecommendation]:
        """Parse AI response and combine with semantic matches"""
        recommendations = []
        
        # Try to parse JSON response
        try:
            if '{' in ai_response and '}' in ai_response:
                json_start = ai_response.find('{')
                json_end = ai_response.rfind('}') + 1
                json_str = ai_response[json_start:json_end]
                data = json.loads(json_str)
                
                for rec in data.get("recommendations", []):
                    recommendations.append(CareerRecommendation(
                        category=rec.get("category", ""),
                        specific_careers=rec.get("specific_careers", []),
                        confidence_score=rec.get("confidence_score", 0),
                        reasoning=rec.get("reasoning", ""),
                        match_score=0.0  # Will be updated with semantic score
                    ))
        except:
            # Fallback parsing if JSON fails
            pass
        
        # If no recommendations from AI, use semantic matches
        if not recommendations:
            category_scores = {}
            for category, career, score in semantic_matches:
                if category not in category_scores:
                    category_scores[category] = {"careers": [], "total_score": 0, "count": 0}
                category_scores[category]["careers"].append(career)
                category_scores[category]["total_score"] += score
                category_scores[category]["count"] += 1
            
            for category, data in category_scores.items():
                avg_score = data["total_score"] / data["count"]
                recommendations.append(CareerRecommendation(
                    category=category,
                    specific_careers=data["careers"][:3],
                    confidence_score=int(avg_score * 100),
                    reasoning=f"Based on semantic analysis of your interests and skills, {category} careers show strong alignment.",
                    match_score=avg_score
                ))
        
        # Sort by confidence score
        recommendations.sort(key=lambda x: x.confidence_score, reverse=True)
        return recommendations[:3]
    
    def generate_career_explanation(self, category: str, specific_career: str, student_interests: str) -> str:
        """Generate detailed explanation for a specific career path"""
        prompt = config.PROMPT_TEMPLATES["career_explanation"].format(
            category=category,
            specific_career=specific_career,
            student_interests=student_interests
        )
        
        return self._call_azure_openai(prompt, max_tokens=800)
    
    def generate_clarifying_questions(self, conversation: str, unclear_areas: List[str]) -> List[str]:
        """Generate clarifying questions when student input is unclear"""
        prompt = config.PROMPT_TEMPLATES["clarifying_questions"].format(
            conversation=conversation,
            unclear_areas=", ".join(unclear_areas)
        )
        
        response = self._call_azure_openai(prompt, max_tokens=500)
        
        # Extract questions from response
        questions = []
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and ('?' in line) and (line[0].isdigit() or line.startswith('-')):
                # Clean up the question
                question = line.split('.', 1)[-1].strip() if '.' in line else line
                question = question.strip(' -1234567890.')
                if question:
                    questions.append(question)
        
        return questions if questions else config.FALLBACK_QUESTIONS["general"]
    
    def add_to_conversation(self, user_input: str, ai_response: str):
        """Add exchange to conversation history"""
        self.conversation_history.append({
            "user": user_input,
            "ai": ai_response,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        })
    
    def get_conversation_text(self) -> str:
        """Get full conversation as text"""
        conversation = ""
        for exchange in self.conversation_history:
            conversation += f"User: {exchange['user']}\nAI: {exchange['ai']}\n\n"
        return conversation
    
    def get_fallback_questions(self, focus_area: Optional[str] = None) -> List[str]:
        """Get fallback questions for specific focus areas"""
        if focus_area and focus_area.lower() in config.FALLBACK_QUESTIONS:
            return config.FALLBACK_QUESTIONS[focus_area.lower()]
        return config.FALLBACK_QUESTIONS["general"] 