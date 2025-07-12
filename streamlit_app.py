import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import List, Dict
import json
from datetime import datetime

from career_advisor import CareerAdvisorAI, ConversationAnalysis, CareerRecommendation
import config

# Set page configuration
st.set_page_config(
    page_title="AI Career Path Advisor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI - Simplified version
st.markdown("""
<style>
    /* Essential styles only */
    .main-header {
        text-align: center;
        color: #1f2937;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    
    .subtitle {
        text-align: center;
        color: #6b7280;
        margin-bottom: 1.5rem;
        font-style: italic;
    }
    
    /* Improved button styling */
    .stButton > button {
        border-radius: 8px;
        border: none;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Clean metric cards */
    [data-testid="metric-container"] {
        background: #f8fafc;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #3b82f6;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #3b82f6, #1d4ed8);
        border-radius: 4px;
    }
    
    /* Expandable sections */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'advisor' not in st.session_state:
    st.session_state.advisor = CareerAdvisorAI()
if 'conversation_started' not in st.session_state:
    st.session_state.conversation_started = False
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None
if 'current_recommendations' not in st.session_state:
    st.session_state.current_recommendations = []
if 'show_questions' not in st.session_state:
    st.session_state.show_questions = False

def main():
    """Main Streamlit application with enhanced UI"""
    
    # Main header with better styling
    st.markdown('<h1 class="main-header">🎯 AI Career Path Advisor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Discover your ideal career path through intelligent conversation analysis and personalized recommendations</p>', unsafe_allow_html=True)
    
    # Sidebar with enhanced design
    with st.sidebar:
        st.markdown("### 🔧 **Configuration**")
        
        # API Configuration status with better visual feedback
        if not config.AZURE_OPENAI_CONFIG["api_key"]:
            st.markdown('''
            <div class="warning-box">
                <h4>⚠️ Setup Required</h4>
                <p>Azure OpenAI API key not configured. Please set up your .env file with Azure OpenAI credentials to get started.</p>
            </div>
            ''', unsafe_allow_html=True)
            return
        else:
            st.markdown('''
            <div class="success-box">
                <h4>✅ Ready to Go!</h4>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Enhanced career paths display
        st.markdown("### 📊 **Available Career Paths**")
        st.markdown(f"*Explore {len(config.CAREER_PATHS)} comprehensive career categories*")
        
        for category, info in config.CAREER_PATHS.items():
            with st.expander(f"🎯 {category} ({len(info['subcategories'])} careers)", expanded=False):
                st.markdown(f"**📝 Description:**")
                st.write(info['description'])
                
                st.markdown(f"**💪 Key Skills Required:**")
                for skill in info['skills'][:4]:  # Show top 4 skills
                    st.markdown(f"• {skill}")
                if len(info['skills']) > 4:
                    st.markdown(f"• *... and {len(info['skills'])-4} more skills*")
                
                st.markdown(f"**🌟 Popular Career Options:**")
                for career in info['subcategories'][:4]:  # Show top 4 careers
                    st.markdown(f"• {career}")
                if len(info['subcategories']) > 4:
                    st.markdown(f"• *... and {len(info['subcategories'])-4} more careers*")
        
        # Add helpful tips section
        st.markdown("---")
        st.markdown("### 💡 **Tips for Best Results**")
        st.markdown("""
        • **Be specific** about your interests and hobbies
        • **Share examples** of activities you enjoy
        • **Mention subjects** you excel in or find fascinating
        • **Describe your ideal** work environment
        • **Talk about your values** and what motivates you
        """)
    
    # Enhanced main content area with better navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 **Start Conversation**", 
        "🔍 **View Analysis**", 
        "🎯 **Get Recommendations**", 
        "📈 **Explore Insights**"
    ])
    
    with tab1:
        conversation_interface()
    
    with tab2:
        analysis_interface()
    
    with tab3:
        recommendations_interface()
    
    with tab4:
        insights_interface()
    
    # Add footer with additional information
    st.markdown("---")
    # st.markdown('''
    # <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); 
    #             border-radius: 16px; margin-top: 2rem;">
    #     <p style="color: #64748b; margin: 0; font-size: 0.9rem;">
    #         💡 <strong>Powered by AI</strong> • Built with Streamlit • 
    #         Your privacy is protected - conversations are not stored permanently
    #     </p>
    # </div>
    # ''', unsafe_allow_html=True)

def conversation_interface():
    """Enhanced interface for conversation with the AI advisor"""
    
    st.markdown("### 💬 **Let's explore your career interests together!**")
    
    # Conversation starter suggestions with better design
    if not st.session_state.conversation_started:
        st.markdown('''
        <div class="info-box">
            <h4>🚀 Ready to discover your perfect career?</h4>
            <p>Share your passions, interests, favorite subjects, or career dreams with me. The more you tell me about yourself, the better I can guide you toward careers that truly match who you are!</p>
            <p><strong>💡 Pro tip:</strong> There are no wrong answers - just be yourself and share what genuinely excites you!</p>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown("**🌟 Quick Start - Choose a conversation starter:**")
        
        # Enhanced conversation starters with better layout
        col1, col2, col3 = st.columns(3)
        
        starters = [
            ("🎨 Creative & Artistic", "I love creative activities like drawing, writing, photography, and design. I enjoy expressing myself through art and creating beautiful things that inspire others."),
            ("🔬 Science & Technology", "I'm fascinated by science, technology, and how things work. I love solving complex problems, conducting experiments, and understanding the world around us."),
            ("👥 People & Relationships", "I'm passionate about helping others and building relationships. I enjoy working with people, making a positive impact, and creating meaningful connections."),
            ("⚽ Sports & Physical", "I'm very active and love sports, fitness, and physical challenges. I enjoy teamwork, competition, and staying healthy and strong."),
            ("📚 Learning & Teaching", "I love learning new things and sharing knowledge with others. I'm curious about many subjects and enjoy explaining concepts to help people understand."),
            ("💼 Business & Leadership", "I'm interested in entrepreneurship, leadership, and making strategic decisions. I enjoy organizing projects and leading teams toward success.")
        ]
        
        # Display starters in a grid
        for i, (title, content) in enumerate(starters):
            col = [col1, col2, col3][i % 3]
            with col:
                if st.button(title, key=f"starter_{i}", use_container_width=True):
                    process_conversation_input(content)
        
        st.markdown("---")
    
    # Enhanced text input area
    st.markdown("**✍️ Share your thoughts with me:**")
    user_input = st.text_area(
        "",
        height=150,
        placeholder="Tell me about:\n• What activities make you lose track of time?\n• Which school subjects do you find most interesting?\n• What kind of problems do you enjoy solving?\n• What values are important to you in a career?\n• Any career ideas you've been curious about?",
        help="💡 The more specific you are, the better I can understand your unique interests and recommend careers that truly fit you.",
        label_visibility="collapsed"
    )
    
    # Enhanced action buttons
    col1, col2, col3, col4 = st.columns([2.5, 2.5, 2.5, 2])
    
    with col1:
        if st.button("💬 **Continue Chat**", type="primary", use_container_width=True):
            if user_input.strip():
                process_conversation_input(user_input)
            else:
                st.warning("💭 Please share something about yourself to continue our conversation!")
    
    with col2:
        if st.button("🔍 **Analyze Now**", use_container_width=True):
            if st.session_state.advisor.conversation_history:
                with st.spinner("🔍 Analyzing your interests..."):
                    analyze_conversation()
            else:
                st.warning("💬 Let's have a conversation first so I can analyze your interests!")
    
    with col3:
        if st.button("❓ **Need Help?**", use_container_width=True):
            st.session_state.show_questions = True
            st.rerun()
    
    with col4:
        if st.button("🔄 **Reset**", use_container_width=True):
            reset_conversation()
            st.rerun()
    
    # Enhanced progress indicator
    if st.session_state.advisor.conversation_history:
        progress = min(len(st.session_state.advisor.conversation_history) / 4, 1.0)
        st.markdown("**📊 Conversation Progress:**")
        st.progress(progress)
        
        # Progress messages
        if progress < 0.3:
            st.caption("🌱 Great start! Keep sharing to get better recommendations.")
        elif progress < 0.7:
            st.caption("🌿 Good progress! I'm learning about your interests.")
        elif progress < 1.0:
            st.caption("🌳 Excellent! I have a solid understanding of your interests.")
        else:
            st.caption("✨ Perfect! Ready for detailed analysis and recommendations.")
        
        st.markdown(f"*{len(st.session_state.advisor.conversation_history)} conversation exchanges completed*")
    
    # Enhanced conversation history display
    if st.session_state.advisor.conversation_history:
        st.markdown("---")
        st.markdown("### 📖 **Our Conversation So Far**")
        
        for i, exchange in enumerate(st.session_state.advisor.conversation_history):
            # User message with enhanced styling
            st.markdown(f'''
            <div class="chat-message user-message">
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <strong style="color: #1e40af; font-size: 1.1rem;">👤 You shared:</strong>
                    <span style="margin-left: auto; color: #64748b; font-size: 0.85rem;">Exchange #{i+1}</span>
                </div>
                <p style="margin: 0; line-height: 1.6;">{exchange["user"]}</p>
            </div>
            ''', unsafe_allow_html=True)
            
            # AI response with enhanced styling
            st.markdown(f'''
            <div class="chat-message ai-message">
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <strong style="color: #6b21a8; font-size: 1.1rem;">🤖 AI Career Advisor:</strong>
                </div>
                <p style="margin: 0; line-height: 1.6;">{exchange["ai"]}</p>
            </div>
            ''', unsafe_allow_html=True)
    
    # Enhanced clarifying questions section
    if st.session_state.show_questions:
        show_enhanced_clarifying_questions()

def reset_conversation():
    """Reset the conversation and start fresh"""
    st.session_state.conversation_started = False
    st.session_state.current_analysis = None
    st.session_state.current_recommendations = []
    st.session_state.show_questions = False
    st.session_state.advisor.conversation_history = []

def process_conversation_input(user_input: str):
    """Process user input and generate AI response"""
    
    with st.spinner("🤔 AI Advisor is thinking..."):
        # Simple acknowledgment response (in production, this would be more sophisticated)
        ai_response = generate_ai_response(user_input)
        
        # Add to conversation history
        st.session_state.advisor.add_to_conversation(user_input, ai_response)
        st.session_state.conversation_started = True
        
        # Check if we need clarifying questions
        if len(st.session_state.advisor.conversation_history) >= 2:
            conversation_text = st.session_state.advisor.get_conversation_text()
            if len(conversation_text.split()) < 50:  # If conversation is too brief
                st.session_state.show_questions = True
    
    st.rerun()

def generate_ai_response(user_input: str) -> str:
    """Generate appropriate AI response based on user input"""
    
    # Simple response generation - in production, this would be more sophisticated
    responses = [
        f"That's interesting! I can see you have some great insights about your interests. Tell me more about what specifically draws you to these areas.",
        f"Thanks for sharing that with me. What aspects of these activities or subjects make you feel most engaged and motivated?",
        f"I appreciate you opening up about your interests. Can you tell me more about your ideal work environment or the type of impact you'd like to make?",
        f"Great! I'm getting a good sense of your interests. What are some challenges or problems you'd be excited to work on solving?"
    ]
    
    # Select response based on conversation length
    response_index = min(len(st.session_state.advisor.conversation_history), len(responses) - 1)
    return responses[response_index]

def show_enhanced_clarifying_questions():
    """Display enhanced clarifying questions to get more information"""
    
    st.markdown("---")
    st.markdown("### 🤔 **Let me ask you a few specific questions**")
    st.markdown('''
    <div class="info-box">
        <p>I'd love to understand you better! Click on any question below that interests you, and I'll help guide our conversation in that direction.</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Get clarifying questions
    conversation_text = st.session_state.advisor.get_conversation_text()
    questions = st.session_state.advisor.generate_clarifying_questions(
        conversation_text, 
        ["interests", "skills", "work_environment", "values", "goals"]
    )
    
    # Enhanced question categories
    question_categories = {
        "🎯 About Your Interests": [
            "What hobbies or activities do you find yourself doing in your free time?",
            "Which school subjects make you excited to learn more?",
            "What topics do you love reading about or watching videos on?"
        ],
        "💪 About Your Strengths": [
            "What do friends and family often ask for your help with?",
            "What skills come naturally to you?",
            "What achievements are you most proud of?"
        ],
        "🏢 About Work Environment": [
            "Do you prefer working alone or with others?",
            "Would you rather work indoors or outdoors?",
            "Do you like routine tasks or prefer variety and new challenges?"
        ],
        "⭐ About Your Values": [
            "What kind of impact do you want to make in the world?",
            "Is work-life balance important to you?",
            "Do you prefer stability or are you comfortable with risk?"
        ]
    }
    
    # Display questions in organized categories
    for category, category_questions in question_categories.items():
        st.markdown(f"#### {category}")
        col1, col2 = st.columns(2)
        for i, question in enumerate(category_questions):
            col = col1 if i % 2 == 0 else col2
            with col:
                if st.button(f"💭 {question}", key=f"cat_question_{category}_{i}", use_container_width=True):
                    process_conversation_input(f"Great question! Let me think about that: {question}")
                    st.session_state.show_questions = False
                    st.rerun()
    
    # Option to continue without questions
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ **I'm ready to analyze**", type="primary", use_container_width=True):
            st.session_state.show_questions = False
            if st.session_state.advisor.conversation_history:
                analyze_conversation()
            st.rerun()
    
    with col2:
        if st.button("❌ **Skip questions**", use_container_width=True):
            st.session_state.show_questions = False
            st.rerun()

def analyze_conversation():
    """Analyze the conversation and extract preferences with enhanced feedback"""
    
    conversation_text = st.session_state.advisor.get_conversation_text()
    
    with st.spinner("🧠 AI is analyzing your conversation and identifying your interests..."):
        analysis = st.session_state.advisor.extract_preferences(conversation_text)
        st.session_state.current_analysis = analysis
        
        # Also generate recommendations
        recommendations = st.session_state.advisor.map_to_career_paths(analysis)
        st.session_state.current_recommendations = recommendations
    
    st.success("🎉 **Great news! Here's what I discovered about you:**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Interests Found", len(analysis.interests))
    with col2:
        st.metric("💪 Skills Identified", len(analysis.skills))
    with col3:
        st.metric("🎯 Confidence Level", f"{int(analysis.confidence_score * 100)}%")
    
    st.rerun()

def analysis_interface():
    """Clean interface for displaying conversation analysis using native Streamlit components"""
    
    if not st.session_state.current_analysis:
        st.info("💬 **Ready for your personalized analysis?**")
        st.write("Start by having a conversation about your interests and goals in the **Start Conversation** tab, then come back here to see detailed insights about your career preferences!")
        st.caption("💡 **Tip:** The more you share, the more accurate your analysis will be.")
        return
    
    analysis = st.session_state.current_analysis
    
    st.header("🔍 Your Personalized Interest Analysis")
    st.caption("*Based on our conversation, here's what I've learned about you:*")
    
    # Metrics using native Streamlit components
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Interests", len(analysis.interests))
        
    with col2:
        st.metric("💪 Skills", len(analysis.skills))
        
    with col3:
        st.metric("📚 Academic Areas", len(analysis.academic_preferences))
        
    with col4:
        confidence_score = int(analysis.confidence_score * 100)
        st.metric("📊 Confidence", f"{confidence_score}%")
    
    st.divider()
    
    # Detailed analysis with clean organization
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Interests section
        if analysis.interests:
            st.subheader("🎯 Your Core Interests")
            st.caption("These are the areas that seem to capture your attention and passion:")
            
            for interest in analysis.interests:
                st.info(f"• {interest}")
        
        # Skills section
        if analysis.skills:
            st.subheader("💪 Your Natural Strengths")
            st.caption("These skills seem to come naturally to you or you've developed them well:")
            
            for skill in analysis.skills:
                st.success(f"• {skill}")
    
    with col2:
        # Academic preferences
        if analysis.academic_preferences:
            st.subheader("📚 Academic Preferences")
            st.caption("Subject areas that align with your interests:")
            
            for pref in analysis.academic_preferences:
                st.warning(f"• {pref}")
        
        # Work environment and values
        if analysis.work_environment or analysis.values:
            st.subheader("🏢 Work Style & Values")
            st.caption("Your preferred work environment and important values:")
            
            for env in analysis.work_environment:
                st.info(f"🏢 {env}")
            
            for value in analysis.values:
                st.error(f"⭐ {value}")
    
    # Career goals section (full width)
    if analysis.career_goals:
        st.subheader("🚀 Your Career Aspirations")
        st.caption("Goals and aspirations you've mentioned:")
        
        for goal in analysis.career_goals:
            st.success(f"🚀 {goal}")
    
    # Next steps call to action
    st.divider()
    st.success("🎯 **Ready for your career recommendations?**")
    st.write("Now that I understand your interests and strengths, head over to the **Get Recommendations** tab to see careers that match your profile!")

def recommendations_interface():
    """Clean interface for displaying career recommendations using native Streamlit components"""
    
    if not st.session_state.current_recommendations:
        st.info("🎯 **Ready for your personalized career recommendations?**")
        st.write("Complete your conversation and analysis first to see careers perfectly matched to your interests, skills, and goals!")
        
        with st.expander("📋 **Next Steps**", expanded=True):
            st.write("1. Go to **Start Conversation** and share your interests")
            st.write("2. Click **Analyze Now** to process your preferences") 
            st.write("3. Return here for your personalized career matches!")
        return
    
    st.header("🎯 Your Personalized Career Recommendations")
    st.caption("*Based on your unique interests, skills, and preferences, here are careers that match your profile:*")
    
    recommendations = st.session_state.current_recommendations
    
    # Summary metrics using native Streamlit metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Career Matches", len(recommendations))
    
    with col2:
        avg_confidence = sum(rec.confidence_score for rec in recommendations) / len(recommendations)
        st.metric("📊 Avg Match", f"{int(avg_confidence)}%")
    
    with col3:
        high_matches = len([r for r in recommendations if r.confidence_score >= 70])
        st.metric("⭐ Strong Matches", high_matches)
    
    with col4:
        total_careers = sum(len(rec.specific_careers or []) for rec in recommendations)
        st.metric("🌟 Total Careers", total_careers)
    
    st.divider()
    
    # Display recommendations using native Streamlit components
    for i, rec in enumerate(recommendations):
        # Determine match quality
        if rec.confidence_score >= 80:
            match_quality = "Excellent Match"
            match_emoji = "⭐⭐⭐"
            color = "green"
        elif rec.confidence_score >= 60:
            match_quality = "Good Match"
            match_emoji = "⭐⭐"
            color = "blue"
        elif rec.confidence_score >= 40:
            match_quality = "Fair Match"
            match_emoji = "⭐"
            color = "orange"
        else:
            match_quality = "Worth Exploring"
            match_emoji = ""
            color = "gray"
        
        # Create expandable section for each recommendation
        with st.expander(f"#{i+1} {rec.category} - {match_quality} {match_emoji}", expanded=i < 2):
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader(f"🎯 {rec.category}")
                st.write(f"**Why this career path fits you:**")
                st.write(rec.reasoning)
                
                # Show specific careers if available
                if rec.specific_careers:
                    st.write("**🌟 Recommended Specific Careers:**")
                    career_columns = st.columns(min(3, len(rec.specific_careers)))
                    for j, career in enumerate(rec.specific_careers[:6]):
                        col_idx = j % 3
                        with career_columns[col_idx]:
                            st.info(f"💼 {career}")
                    
                    if len(rec.specific_careers) > 6:
                        st.caption(f"*... and {len(rec.specific_careers) - 6} more careers in this field*")
            
            with col2:
                # Confidence score display
                st.metric("Match Score", f"{rec.confidence_score}%")
                
                # Progress bar for visual representation
                progress_value = rec.confidence_score / 100
                st.progress(progress_value)
                
                # Color-coded badge based on score
                if rec.confidence_score >= 80:
                    st.success("Highly Recommended")
                elif rec.confidence_score >= 60:
                    st.info("Recommended")
                elif rec.confidence_score >= 40:
                    st.warning("Worth Exploring")
                else:
                    st.error("Research Further")
            
            # Action buttons using native Streamlit buttons
            st.write("**📋 Actions:**")
            action_col1, action_col2, action_col3 = st.columns(3)
            
            with action_col1:
                if st.button(f"📖 Learn More", key=f"learn_more_{i}"):
                    show_career_explanation(rec)
            
            with action_col2:
                if st.button(f"🔍 Explore Careers", key=f"explore_{i}"):
                    show_career_details(rec, i)
            
            with action_col3:
                if st.button(f"📊 Compare Options", key=f"compare_{i}"):
                    show_career_comparison(rec, i)
    
    # Call to action section
    st.divider()
    st.success("🎉 **Ready to take the next step in your career journey?**")
    st.write("Explore detailed insights and download your personalized career report in the **Explore Insights** tab!")
    
    # Quick navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📈 View Insights & Analytics", type="primary", use_container_width=True):
            st.switch_page("insights")  # This would need proper page navigation
    
    with col2:
        if st.button("💬 Continue Conversation", use_container_width=True):
            st.switch_page("conversation")  # This would need proper page navigation

def show_career_details(recommendation: CareerRecommendation, index: int):
    """Show detailed information about specific careers"""
    
    st.markdown("---")
    st.markdown(f"### 📋 **Detailed Career Information: {recommendation.category}**")
    
    # Get category information
    category_info = config.CAREER_PATHS.get(recommendation.category, {})
    
    if category_info:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📝 Field Description:**")
            st.write(category_info['description'])
            
            st.markdown("**💪 Essential Skills:**")
            for skill in category_info['skills']:
                st.markdown(f"• {skill}")
        
        with col2:
            st.markdown("**🌟 Career Opportunities:**")
            
            # Show careers in expandable sections
            careers_per_section = 8
            for i in range(0, len(category_info['subcategories']), careers_per_section):
                section_careers = category_info['subcategories'][i:i+careers_per_section]
                section_title = f"Careers {i+1}-{min(i+careers_per_section, len(category_info['subcategories']))}"
                
                with st.expander(section_title):
                    for career in section_careers:
                        st.markdown(f"• **{career}**")

def show_career_comparison(recommendation: CareerRecommendation, index: int):
    """Show comparison between career options"""
    
    st.markdown("---")
    st.markdown(f"### ⚖️ **Career Comparison: {recommendation.category}**")
    
    if recommendation.specific_careers:
        comparison_data = []
        
        for career in recommendation.specific_careers[:5]:  # Compare top 5 careers
            # This would ideally use more sophisticated career data
            comparison_data.append({
                'Career': career,
                'Growth Outlook': ['High 📈', 'Medium 📊', 'Stable 📉'][hash(career) % 3],
                'Education Required': ['Bachelor\'s', 'Master\'s', 'Certification'][hash(career) % 3],
                'Work Environment': ['Office', 'Remote', 'Field Work'][hash(career) % 3],
                'Match Score': f"{recommendation.confidence_score - (hash(career) % 10)}%"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

def show_career_explanation(recommendation: CareerRecommendation):
    """Show detailed explanation for a career recommendation"""
    
    st.markdown("---")
    st.subheader(f"📖 Deep Dive: {recommendation.category} Careers")
    
    # Get student interests for personalized explanation
    interests_text = ""
    if st.session_state.current_analysis:
        analysis = st.session_state.current_analysis
        interests_text = f"Interests: {', '.join(analysis.interests)}, Skills: {', '.join(analysis.skills)}"
    
    with st.spinner("Generating personalized career explanation..."):
        explanation = st.session_state.advisor.generate_career_explanation(
            recommendation.category,
            recommendation.specific_careers[0] if recommendation.specific_careers else recommendation.category,
            interests_text
        )
    
    st.write(explanation)
    
    # Show related careers in this category
    category_info = config.CAREER_PATHS.get(recommendation.category, {})
    if category_info:
        st.subheader(f"🌟 All {recommendation.category} Career Options")
        
        careers_df = pd.DataFrame({
            'Career': category_info['subcategories'],
            'Category': [recommendation.category] * len(category_info['subcategories'])
        })
        
        st.dataframe(careers_df, use_container_width=True)

def insights_interface():
    """Enhanced interface for displaying insights and analytics"""
    
    if not st.session_state.current_recommendations:
        st.markdown('''
        <div class="info-box">
            <h4>📊 Ready to explore your career insights?</h4>
            <p>Complete your career analysis to unlock detailed insights, interactive visualizations, and downloadable reports!</p>
            <p><strong>What you'll get here:</strong></p>
            <ul>
                <li>📈 Interactive career match visualizations</li>
                <li>📊 Detailed comparison tables</li>
                <li>💾 Downloadable career reports</li>
                <li>🎯 Personalized next steps</li>
            </ul>
        </div>
        ''', unsafe_allow_html=True)
        return
    
    st.markdown("### 📈 **Your Career Journey Insights**")
    st.markdown("*Dive deep into your career matches with interactive visualizations and detailed analytics*")
    
    recommendations = st.session_state.current_recommendations
    
    # Enhanced overview metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_categories = len(recommendations)
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value" style="color: #667eea;">{total_categories}</div>
            <div class="metric-label">🎯 Career Fields</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        top_match = max(rec.confidence_score for rec in recommendations)
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value" style="color: #10b981;">{int(top_match)}%</div>
            <div class="metric-label">⭐ Best Match</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        avg_match = sum(rec.confidence_score for rec in recommendations) / len(recommendations)
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value" style="color: #3b82f6;">{int(avg_match)}%</div>
            <div class="metric-label">📊 Average Match</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        strong_matches = len([r for r in recommendations if r.confidence_score >= 70])
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value" style="color: #f59e0b;">{strong_matches}</div>
            <div class="metric-label">🌟 Strong Matches</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col5:
        conversation_exchanges = len(st.session_state.advisor.conversation_history) if st.session_state.advisor.conversation_history else 0
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value" style="color: #8b5cf6;">{conversation_exchanges}</div>
            <div class="metric-label">💬 Conversations</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Enhanced visualizations section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Enhanced confidence scores bar chart
        categories = [rec.category for rec in recommendations]
        scores = [rec.confidence_score for rec in recommendations]
        
        # Create color scale based on scores
        colors = ['#10b981' if score >= 80 else '#3b82f6' if score >= 60 else '#f59e0b' if score >= 40 else '#64748b' for score in scores]
        
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=scores,
                text=[f'{score}%' for score in scores],
                textposition='outside',
                marker=dict(
                    color=colors,
                    line=dict(color='rgba(255,255,255,0.8)', width=2)
                ),
                hovertemplate='<b>%{x}</b><br>Match Score: %{y}%<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=dict(
                text="🎯 Career Path Confidence Scores",
                font=dict(size=16, color='#1e293b', family='Arial, sans-serif'),
                x=0.5
            ),
            xaxis=dict(
                title="Career Categories",
                tickangle=-45,
                gridcolor='rgba(0,0,0,0.1)',
                showgrid=True
            ),
            yaxis=dict(
                title="Confidence Score (%)",
                range=[0, 100],
                gridcolor='rgba(0,0,0,0.1)',
                showgrid=True
            ),
            height=450,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            margin=dict(t=60, b=100, l=60, r=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Enhanced pie chart with better styling
        fig = go.Figure(data=[
            go.Pie(
                labels=categories,
                values=scores,
                hole=0.4,
                marker=dict(
                    colors=colors,
                    line=dict(color='white', width=2)
                ),
                textinfo='label+percent',
                textposition='outside',
                hovertemplate='<b>%{label}</b><br>Match Score: %{value}%<br>Share: %{percent}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=dict(
                text="📊 Career Interest Distribution",
                font=dict(size=16, color='#1e293b', family='Arial, sans-serif'),
                x=0.5
            ),
            height=450,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=60, b=20, l=20, r=20),
            annotations=[
                dict(
                    text=f"Avg: {int(avg_match)}%",
                    x=0.5, y=0.5,
                    font_size=16,
                    font_color='#667eea',
                    showarrow=False
                )
            ]
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Enhanced detailed comparison table
    st.markdown("### 📋 **Comprehensive Career Analysis**")
    
    comparison_data = []
    for i, rec in enumerate(recommendations):
        # Determine match quality and recommendation
        if rec.confidence_score >= 80:
            match_quality = 'Excellent ⭐⭐⭐'
            recommendation_text = 'Highly Recommended'
            priority = 'High Priority'
        elif rec.confidence_score >= 60:
            match_quality = 'Good ⭐⭐'
            recommendation_text = 'Recommended'
            priority = 'Medium Priority'
        elif rec.confidence_score >= 40:
            match_quality = 'Fair ⭐'
            recommendation_text = 'Worth Exploring'
            priority = 'Low Priority'
        else:
            match_quality = 'Consider'
            recommendation_text = 'Explore Further'
            priority = 'Research Needed'
        
        comparison_data.append({
            'Rank': f"#{i + 1}",
            'Career Field': rec.category,
            'Match Score': rec.confidence_score,
            'Match Quality': match_quality,
            'Recommendation': recommendation_text,
            'Priority Level': priority,
            'Specific Careers Available': len(rec.specific_careers) if rec.specific_careers else 0,
            'Next Steps': 'Learn More → Explore → Plan'
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Match Score": st.column_config.ProgressColumn(
                "Match Score (%)",
                help="How well this career field matches your profile",
                min_value=0,
                max_value=100,
                format="%d%%",
            ),
            "Specific Careers Available": st.column_config.NumberColumn(
                "Available Careers",
                help="Number of specific career options in this field",
                min_value=0,
                max_value=50,
            ),
        }
    )
    
    # Action plan section
    st.markdown("### 🎯 **Your Personalized Action Plan**")
    
    # Generate action plan based on top recommendations
    top_recommendation = recommendations[0] if recommendations else None
    
    if top_recommendation:
        st.subheader(f"🚀 Recommended Next Steps for {top_recommendation.category}")
        
        # Create clean action steps using native Streamlit components
        action_steps = [
            {
                "number": "1",
                "title": "Research & Explore", 
                "description": f"Learn more about careers in {top_recommendation.category}. Read job descriptions, salary ranges, and day-to-day responsibilities.",
                "color": "blue"
            },
            {
                "number": "2", 
                "title": "Skill Development",
                "description": "Identify key skills needed and start building them through courses, projects, or volunteer opportunities.",
                "color": "green"
            },
            {
                "number": "3",
                "title": "Network & Connect", 
                "description": "Connect with professionals in this field through LinkedIn, informational interviews, or industry events.",
                "color": "orange"
            },
            {
                "number": "4",
                "title": "Gain Experience",
                "description": "Look for internships, part-time jobs, or project opportunities to get hands-on experience in this field.",
                "color": "red"
            }
        ]
        
        for step in action_steps:
            with st.container():
                col1, col2 = st.columns([1, 10])
                
                with col1:
                    if step["color"] == "blue":
                        st.info(step["number"])
                    elif step["color"] == "green": 
                        st.success(step["number"])
                    elif step["color"] == "orange":
                        st.warning(step["number"]) 
                    else:
                        st.error(step["number"])
                
                with col2:
                    st.write(f"**{step['title']}**")
                    st.write(step["description"])
                
                st.write("")  # Add spacing
    
    # Enhanced export and next steps section
    st.markdown("### 💾 **Export & Share Your Results**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📄 **Career Report**", use_container_width=True):
            report_data = generate_enhanced_report()
            st.download_button(
                label="📥 Download Complete Report (JSON)",
                data=json.dumps(report_data, indent=2),
                file_name=f"career_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    with col2:
        if st.button("📊 **Data Export**", use_container_width=True):
            csv_data = comparison_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Analysis (CSV)",
                data=csv_data,
                file_name=f"career_matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col3:
        if st.button("📝 **Action Plan**", use_container_width=True):
            action_plan = generate_action_plan_text()
            st.download_button(
                label="📥 Download Action Plan (TXT)",
                data=action_plan,
                file_name=f"career_action_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    with col4:
        if st.button("🔄 **Start Over**", use_container_width=True):
            if st.button("✅ Confirm Reset", key="confirm_reset"):
                reset_conversation()
                st.rerun()
    
    # Success message and next steps
    st.divider()
    st.success("🎉 **Congratulations on completing your career analysis!**")
    st.write("You now have a comprehensive understanding of career paths that align with your interests and strengths. Remember, this is just the beginning of your career journey!")
    st.info("💡 **Pro tip:** Career exploration is an ongoing process. Come back anytime to reassess your interests as you grow and learn more about yourself.")

def generate_enhanced_report() -> Dict:
    """Generate a comprehensive enhanced report of the analysis"""
    
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "report_version": "2.0",
            "analysis_type": "AI Career Path Advisor"
        },
        "user_profile": {},
        "career_analysis": {},
        "recommendations": [],
        "action_plan": {},
        "conversation_summary": {}
    }
    
    if st.session_state.current_analysis:
        analysis = st.session_state.current_analysis
        report["user_profile"] = {
            "interests": analysis.interests,
            "academic_preferences": analysis.academic_preferences,
            "skills": analysis.skills,
            "work_environment_preferences": analysis.work_environment,
            "values": analysis.values,
            "career_goals": analysis.career_goals
        }
        
        report["career_analysis"] = {
            "confidence_score": analysis.confidence_score,
            "analysis_quality": "High" if analysis.confidence_score > 0.7 else "Medium" if analysis.confidence_score > 0.5 else "Needs More Data",
            "total_data_points": {
                "interests": len(analysis.interests),
                "skills": len(analysis.skills),
                "academic_areas": len(analysis.academic_preferences),
                "work_preferences": len(analysis.work_environment),
                "values": len(analysis.values),
                "goals": len(analysis.career_goals)
            }
        }
    
    if st.session_state.current_recommendations:
        for i, rec in enumerate(st.session_state.current_recommendations):
            report["recommendations"].append({
                "rank": i + 1,
                "category": rec.category,
                "specific_careers": rec.specific_careers,
                "confidence_score": rec.confidence_score,
                "match_quality": "Excellent" if rec.confidence_score >= 80 else "Good" if rec.confidence_score >= 60 else "Fair",
                "reasoning": rec.reasoning,
                "recommended_action": "Highly Recommended" if rec.confidence_score >= 80 else "Recommended" if rec.confidence_score >= 60 else "Worth Exploring"
            })
    
    # Generate action plan
    if st.session_state.current_recommendations:
        top_rec = st.session_state.current_recommendations[0]
        report["action_plan"] = {
            "primary_focus": top_rec.category,
            "immediate_steps": [
                f"Research careers in {top_rec.category}",
                "Identify required skills and qualifications",
                "Connect with professionals in the field",
                "Explore educational pathways"
            ],
            "long_term_goals": [
                "Develop relevant skills through courses or projects",
                "Gain practical experience through internships",
                "Build a professional network",
                "Create a career development plan"
            ]
        }
    
    # Conversation summary
    if st.session_state.advisor.conversation_history:
        report["conversation_summary"] = {
            "total_exchanges": len(st.session_state.advisor.conversation_history),
            "conversation_quality": "High" if len(st.session_state.advisor.conversation_history) >= 3 else "Medium",
            "key_topics_discussed": ["interests", "skills", "preferences", "goals"],
            "conversation_history": st.session_state.advisor.conversation_history
        }
    
    return report

def generate_action_plan_text() -> str:
    """Generate a text-based action plan"""
    
    action_plan = "🎯 PERSONALIZED CAREER ACTION PLAN\n"
    action_plan += "=" * 50 + "\n\n"
    action_plan += f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}\n\n"
    
    if st.session_state.current_recommendations:
        top_rec = st.session_state.current_recommendations[0]
        action_plan += f"PRIMARY CAREER FOCUS: {top_rec.category}\n"
        action_plan += f"Confidence Score: {top_rec.confidence_score}%\n\n"
        
        action_plan += "WHY THIS FITS YOU:\n"
        action_plan += f"{top_rec.reasoning}\n\n"
        
        action_plan += "IMMEDIATE NEXT STEPS (Next 30 Days):\n"
        action_plan += f"□ Research 3-5 specific careers in {top_rec.category}\n"
        action_plan += "□ Read job descriptions and salary information\n"
        action_plan += "□ Identify 2-3 professionals to connect with on LinkedIn\n"
        action_plan += "□ Look for relevant online courses or certifications\n\n"
        
        action_plan += "SHORT-TERM GOALS (Next 3-6 Months):\n"
        action_plan += "□ Complete at least one relevant skill-building course\n"
        action_plan += "□ Conduct 2-3 informational interviews\n"
        action_plan += "□ Start a project related to this field\n"
        action_plan += "□ Join professional groups or communities\n\n"
        
        action_plan += "LONG-TERM OBJECTIVES (6+ Months):\n"
        action_plan += "□ Apply for internships or entry-level positions\n"
        action_plan += "□ Build a portfolio showcasing relevant skills\n"
        action_plan += "□ Attend industry events or conferences\n"
        action_plan += "□ Develop a 5-year career plan\n\n"
        
        if top_rec.specific_careers:
            action_plan += "SPECIFIC CAREER OPTIONS TO EXPLORE:\n"
            for career in top_rec.specific_careers[:5]:
                action_plan += f"• {career}\n"
            action_plan += "\n"
    
    if st.session_state.current_analysis:
        analysis = st.session_state.current_analysis
        action_plan += "YOUR KEY STRENGTHS TO LEVERAGE:\n"
        for skill in analysis.skills:
            action_plan += f"• {skill}\n"
        action_plan += "\n"
        
        action_plan += "AREAS OF INTEREST TO DEVELOP:\n"
        for interest in analysis.interests:
            action_plan += f"• {interest}\n"
        action_plan += "\n"
    
    action_plan += "RESOURCES FOR CONTINUED EXPLORATION:\n"
    action_plan += "• LinkedIn Learning for skill development\n"
    action_plan += "• Indeed or Glassdoor for job research\n"
    action_plan += "• Professional association websites\n"
    action_plan += "• University career services\n"
    action_plan += "• Industry-specific forums and communities\n\n"
    
    action_plan += "Remember: Career exploration is a journey, not a destination. Stay curious and open to new opportunities!\n"
    
    return action_plan

if __name__ == "__main__":
    main() 