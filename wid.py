import streamlit as st
import PyPDF2
import pytesseract
from PIL import Image
import pdf2image
from groq import Groq
import os
from io import BytesIO
import tempfile
import re
from datetime import datetime
import json

st.set_page_config(
    page_title="SimpleTech ",
    page_icon="🤖",
    layout="wide"
)

if 'simplified_content' not in st.session_state:
    st.session_state.simplified_content = ""
if 'original_content' not in st.session_state:
    st.session_state.original_content = ""
if 'translated_content' not in st.session_state:
    st.session_state.translated_content = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_settings' not in st.session_state:
    st.session_state.current_settings = {}
if 'file_info' not in st.session_state:
    st.session_state.file_info = {}
if 'processing_history' not in st.session_state:
    st.session_state.processing_history = []
if 'user_feedback' not in st.session_state:
    st.session_state.user_feedback = ""
if 'conversation_mode' not in st.session_state:
    st.session_state.conversation_mode = False

class EnhancedTechnicalSimplifier:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.client = None
        if api_key:
            self.client = Groq(api_key=api_key)

    def extract_text_from_image(self, image_file):
        """Extract text from image files using OCR"""
        try:
            image = Image.open(image_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            text = pytesseract.image_to_string(image)
            return text, 1
        except Exception as e:
            st.error(f"Image OCR failed: {str(e)}")
            return "", 0

    # Using PyPDF2 first because it faster than ORC
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF using PyPDF2 or OCR fallback"""
        text = ""
        page_count = 0

        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            page_count = len(pdf_reader.pages)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text.strip():
                    text += page_text + "\n"

            if len(text.strip()) > 100:
                return text, page_count
        except Exception as e:
            st.warning(f"PyPDF2 extraction failed: {str(e)}")

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                images = pdf2image.convert_from_path(tmp_file_path)
                page_count = len(images)
                ocr_text = ""
                for i, image in enumerate(images):
                    try:
                        page_text = pytesseract.image_to_string(image)
                        if page_text.strip():
                            ocr_text += f"\n--- Page {i+1} ---\n{page_text}\n"
                    except Exception as page_error:
                        st.warning(f"Failed to OCR page {i+1}: {str(page_error)}")
                text = ocr_text
            except Exception as conversion_error:
                if "poppler" in str(conversion_error).lower():
                    st.error("""**Poppler is not installed!** Please install Poppler or try uploading an image file instead.""")
                else:
                    st.error(f"PDF to image conversion failed: {str(conversion_error)}")
            finally:
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)

            return text, page_count
        except Exception as e:
            st.error(f"PDF processing failed: {str(e)}")
            return "", 0

    def extract_text_from_file(self, uploaded_file):
        """Universal text extraction method"""
        file_extension = uploaded_file.name.lower().split('.')[-1]

        if file_extension == 'pdf':
            text, page_count = self.extract_text_from_pdf(uploaded_file)
            return text, page_count, 'pdf'
        elif file_extension in ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif']:
            text, page_count = self.extract_text_from_image(uploaded_file)
            return text, page_count, 'image'
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return "", 0, 'unknown'

    def clean_text(self, text):
        """Clean and preprocess extracted text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\d+\n', '\n', text)
        text = re.sub(r'\n--- Page \d+ ---\n', '\n', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()

    def build_context_from_history(self):
        """Build context from chat history for better responses"""
        context = ""
        if st.session_state.chat_history:
            context += "\n\nPrevious conversation context:\n"
            for entry in st.session_state.chat_history[-3:]:  # Last 3 exchanges
                if entry['type'] == 'user':
                    context += f"User: {entry['content']}\n"
                elif entry['type'] == 'assistant':
                    context += f"Assistant: {entry['content'][:200]}...\n"
        return context

    def simplify_content_with_chat(self, text, complexity_level="beginner", focus_area="general",
                                 model_choice="llama3-8b-8192", user_prompt="", is_followup=False):
        """Enhanced simplification with chat context and user feedback"""
        if not self.client:
            return self._mock_simplification(text, complexity_level)

        # Build conversation context
        context = self.build_context_from_history()

        # Define base prompts
        base_prompts = {
            "beginner": """
                Role: Simplified Content Expert (Beginner Level)
                Profile:
                - Language: English
                - Description: Explain complex technical content in the most accessible way possible to complete beginners
                - Background: Expert in breaking down agricultural and technical concepts for general public
                - Personality: Extremely patient, encouraging, and empathetic
                - Target Audience: General public, new farmers, individuals with zero technical background

                Skills & Approach:
                - Break down complex concepts into the simplest possible terms
                - Use everyday analogies (cooking, gardening, household tasks) to explain technical processes
                - Eliminate ALL technical jargon and abbreviations
                - Use storytelling techniques to make content engaging
                - Focus on practical, real-world applications

                Rules:
                - Use only non-technical vocabulary with simple definitions
                - Apply consistent analogies throughout the explanation
                - Remain empathetic and encouraging
                - Focus on "why this matters" for everyday life
                - Avoid complex sentences or overwhelming details

                Workflow for Response:
                1. Create a friendly, encouraging opening
                2. Provide a simple summary (2-3 sentences) using everyday language
                3. Break content into digestible parts with clear analogies
                4. Include practical tips and beginner-friendly warnings
                5. End with encouragement and next steps

                Structure your response with:
                - 🌱 **What This Means for You**: Simple 2-3 sentence summary
                - 🔍 **Breaking It Down**: Key concepts explained with analogies
                - 💡 **Practical Tips**: Beginner-friendly advice and warnings
                - 🚀 **Your Next Steps**: Clear, actionable guidance
            """,
            "intermediate": """
                Role: Simplified Content Expert (Intermediate Level)
                Profile:
                - Language: English with selective technical terms
                - Description: Simplify technical content for users with basic agricultural or technical exposure
                - Background: Expert in bridging basic knowledge to practical implementation
                - Personality: Approachable, practical, and solution-focused
                - Target Audience: Small-scale farmers, agricultural students, tech-curious individuals

                Skills & Approach:
                - Maintain technical accuracy while ensuring clarity
                - Use agricultural analogies and real-world farming examples
                - Introduce technical terms with clear explanations
                - Focus on practical implementation and actionable insights
                - Balance simplicity with useful detail

                Rules:
                - Use moderately technical terms with clear definitions
                - Apply farming and agricultural analogies consistently
                - Provide practical implementation guidance
                - Include considerations for small-scale applications
                - Encourage critical thinking and questions

                Workflow for Response:
                1. Acknowledge the reader's existing knowledge
                2. Provide a comprehensive summary (3-4 sentences)
                3. Explain concepts with agricultural context and examples
                4. Include implementation considerations and best practices
                5. Suggest resources for deeper learning

                Structure your response with:
                - 📋 **Summary & Context**: 3-4 sentence overview with purpose
                - 🔧 **How It Works**: Technical concepts explained with farming analogies
                - ⚙️ **Implementation Guide**: Practical steps and considerations
                - 🎯 **Best Practices**: Proven approaches and common pitfalls to avoid
                - 📚 **Going Deeper**: Suggestions for advanced learning
            """,
            "advanced": """
                Role: Simplified Content Expert (Advanced Level)
                Profile:
                - Language: Technical English with professional terminology
                - Description: Refine and enhance technical content for agricultural professionals and developers
                - Background: Expert in agricultural technology, IoT systems, and professional implementation
                - Personality: Professional, precise, and innovation-focused
                - Target Audience: Agricultural engineers, tech developers, agribusiness professionals, researchers

                Skills & Approach:
                - Preserve technical accuracy while improving clarity and structure
                - Integrate agricultural IoT and modern technology perspectives
                - Focus on scalability, efficiency, and professional applications
                - Provide strategic insights and advanced implementation guidance
                - Connect concepts to broader agricultural technology ecosystem

                Rules:
                - Use appropriate technical terminology with context
                - Maintain professional accuracy and precision
                - Focus on scalable and sustainable solutions
                - Include regulatory, security, and compliance considerations
                - Encourage innovation and advanced applications

                Workflow for Response:
                1. Establish technical context and scope
                2. Provide detailed technical summary (4-5 sentences)
                3. Analyze concepts with professional agricultural technology perspective
                4. Address scalability, security, and implementation challenges
                5. Suggest advanced applications and future considerations

                Structure your response with:
                - 🎯 **Technical Overview**: Detailed 4-5 sentence summary with objectives and scope
                - 🔬 **Technical Analysis**: In-depth explanation with agricultural IoT integration
                - 🏗️ **Implementation Strategy**: Scalability, architecture, and deployment considerations
                - ⚡ **Critical Factors**: Security, compliance, regulations, and risk management
                - 🚀 **Advanced Applications**: Innovation opportunities and future development paths
                - 📊 **Success Metrics**: KPIs and measurement strategies for professional implementation
            """
        }

        if is_followup and user_prompt:
            # Handle follow-up conversation
            system_prompt = f"""You are a helpful AI assistant specializing in technical content simplification.

            The user is asking for modifications to previously simplified content based on their feedback.

            Previous simplification context:
            - Complexity level: {complexity_level}
            - Focus area: {focus_area}

            {context}

            Original content being discussed:
            {text[:2000]}...

            Please respond to the user's specific request while maintaining the overall simplification approach."""

            user_message = f"User feedback/request: {user_prompt}"

        else:
            # Initial simplification
            focus_prompts = {
                "general": "",
                "security": "Focus on security implications and best practices.",
                "development": "Emphasize development practices and implementation details.",
                "business": "Highlight business impact and practical applications.",
                "operations": "Focus on operational procedures and maintenance aspects."
            }

            system_prompt = f"""{base_prompts[complexity_level]}

            {focus_prompts[focus_area]}

            Structure your response clearly with headings and bullet points where appropriate.
            Make the content engaging and actionable.

            {context}"""

            user_message = f"Please simplify this technical content for a {complexity_level} audience:\n\n{text[:4000]}"

            if user_prompt:
                user_message += f"\n\nAdditional user requirements: {user_prompt}"

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                model=model_choice or "llama3-8b-8192",
                max_tokens=2000,
                temperature=0.3
            )

            response = chat_completion.choices[0].message.content

            # Save to chat history
            self.add_to_chat_history("user", user_prompt if user_prompt else "Simplify this content", complexity_level, focus_area)
            self.add_to_chat_history("assistant", response, complexity_level, focus_area)

            return response

        except Exception as e:
            st.error(f"AI processing failed: {str(e)}")
            return self._mock_simplification(text, complexity_level)

    def add_to_chat_history(self, role, content, complexity_level=None, focus_area=None):
        """Add interaction to chat history with metadata"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": role,
            "content": content,
            "complexity_level": complexity_level,
            "focus_area": focus_area
        }
        st.session_state.chat_history.append(entry)

    def _mock_simplification(self, text, complexity_level):
        """Mock simplification when AI is not available"""
        word_count = len(text.split())
        return f"""
        📋 **Summary** (Mock Response - AI not configured)
        This document contains approximately {word_count} words of technical content.

        🔑 **Key Points Identified:**
        - Technical documentation detected
        - Content complexity: {complexity_level} level
        - Automated analysis would break down complex terms
        - Explanations would be tailored to your selected audience

        💡 **What AI Simplification Would Provide:**
        - Plain English explanations of technical terms
        - Step-by-step breakdowns of complex processes
        - Real-world analogies to make concepts clearer
        - Practical implications and applications

        ⚙️ **To Enable Full AI Features:**
        Configure your Groq API key to get actual content simplification and chat features.

        **Original Content Preview:**
        {text[:500]}...
        """

    def translate_text(self, text, target_language="vi", model_choice="llama3-8b-8192"):
        """Translate text with context awareness"""
        if not self.client:
            return f"[Translation unavailable - missing API key]\n\n{text}"

        system_prompt = f"""Translate the following content into {target_language}.
        Maintain the formatting, structure, and technical accuracy.
        If there are technical terms, provide appropriate translations or keep them in English with explanations if needed."""

        try:
            result = self.client.chat.completions.create(
                model=model_choice,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text[:4000]}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            return result.choices[0].message.content
        except Exception as e:
            st.error(f"Translation failed: {str(e)}")
            return text


def main():
    st.title("🤖 AI Technical Content Simplifier")
    st.markdown("Transform complex technical documents with AI-powered conversation and memory")

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    api_key = st.secrets["GROQ_API_KEY"]

    complexity_level = st.sidebar.selectbox(
        "Target Audience Level",
        ["beginner", "intermediate", "advanced"],
        help="Choose the technical level of your audience"
    )

    model_choice = st.sidebar.selectbox(
        "AI Model",
        ["llama3-8b-8192", "mixtral-8x7b-32768", "llama3-70b-8192", "deepseek-r1-distill-llama-70b", "gemma2-9b-it"],
        index=0,
        help="Choose the AI model for content processing"
    )

    focus_area = st.sidebar.selectbox(
        "Focus Area",
        ["general", "security", "development", "business", "operations"],
        help="Select the main area of focus"
    )

    available_languages = {
        "None (Original)": None,
        "Vietnamese": "vi",
        "Japanese": "ja",
        "Korean": "ko",
        "Chinese (Simplified)": "zh",
        "French": "fr",
        "German": "de",
        "Spanish": "es"
    }

    translate_to = st.sidebar.selectbox(
        "🌐 Translate To:",
        list(available_languages.keys())
    )

    # Sidebar memory and history
    st.sidebar.markdown("---")
    st.sidebar.header("🧠 Memory & History")

    if st.sidebar.button("🗑️ Clear Memory"):
        st.session_state.chat_history = []
        st.session_state.simplified_content = ""
        st.session_state.translated_content = ""
        st.session_state.conversation_mode = False
        st.rerun()

    if st.session_state.chat_history:
        st.sidebar.markdown(f"**Conversation entries:** {len(st.session_state.chat_history)}")

        with st.sidebar.expander("📜 Recent History"):
            for i, entry in enumerate(st.session_state.chat_history[-5:]):
                role_icon = "👤" if entry['type'] == 'user' else "🤖"
                timestamp = datetime.fromisoformat(entry['timestamp']).strftime("%H:%M")
                st.sidebar.text(f"{role_icon} {timestamp}: {entry['content'][:50]}...")

    # Initialize simplifier
    simplifier = EnhancedTechnicalSimplifier(api_key)

    # Store current settings
    st.session_state.current_settings = {
        'complexity_level': complexity_level,
        'model_choice': model_choice,
        'focus_area': focus_area,
        'translate_to': translate_to
    }

    # Main interface
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📄 Upload & Process")

        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif'],
            help="Upload a technical document or image to simplify"
        )

        if uploaded_file is not None:
            file_type = uploaded_file.name.lower().split('.')[-1]
            st.success(f"📁 File: {uploaded_file.name} ({file_type.upper()})")

            # Store file info
            st.session_state.file_info = {
                'name': uploaded_file.name,
                'type': file_type,
                'size': uploaded_file.size
            }

            # Show preview for images
            if file_type in ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif']:
                with st.expander("📸 Image Preview"):
                    st.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)

            with st.spinner("🔍 Extracting text..."):
                raw_text, page_count, detected_file_type = simplifier.extract_text_from_file(uploaded_file)
                cleaned_text = simplifier.clean_text(raw_text)
                st.session_state.original_content = cleaned_text

                word_count = len(cleaned_text.split())
                char_count = len(cleaned_text)

                if detected_file_type == 'image':
                    st.info(f"🖼️ Image processed | 📊 {word_count} words, {char_count} characters")
                else:
                    st.info(f"📄 {page_count} pages | 📊 {word_count} words, {char_count} characters")

                with st.expander("📝 Extracted Text Preview"):
                    if cleaned_text:
                        st.text_area("Content",
                                   cleaned_text[:1000] + "..." if len(cleaned_text) > 1000 else cleaned_text,
                                   height=200, key="extracted_preview")
                    else:
                        st.warning("⚠️ No text found. Try a different file or check image quality.")

            # Initial processing section
            st.markdown("### 🚀 Initial Processing")

            # Custom prompt for initial processing
            initial_prompt = st.text_area(
                "💬 Additional Instructions (Optional)",
                placeholder="e.g., 'Focus on security aspects', 'Explain with farming analogies', 'Make it very practical'...",
                height=80,
                key="initial_prompt"
            )

            col_process1, col_process2 = st.columns(2)

            with col_process1:
                if st.button("🚀 Simplify Content", type="primary", use_container_width=True):
                    if cleaned_text:
                        with st.spinner("🤖 AI is processing..."):
                            simplified = simplifier.simplify_content_with_chat(
                                cleaned_text,
                                complexity_level,
                                focus_area,
                                model_choice,
                                initial_prompt,
                                is_followup=False
                            )
                            st.session_state.simplified_content = simplified
                            st.session_state.conversation_mode = True

                            if available_languages[translate_to]:
                                with st.spinner("🌐 Translating..."):
                                    translated = simplifier.translate_text(
                                        simplified,
                                        target_language=available_languages[translate_to],
                                        model_choice=model_choice
                                    )
                                    st.session_state.translated_content = translated
                            else:
                                st.session_state.translated_content = ""
                    else:
                        st.error("❌ No text content found.")

            with col_process2:
                if st.button("🔄 Try Again", use_container_width=True):
                    if st.session_state.original_content:
                        with st.spinner("🤖 Regenerating..."):
                            simplified = simplifier.simplify_content_with_chat(
                                st.session_state.original_content,
                                complexity_level,
                                focus_area,
                                model_choice,
                                initial_prompt,
                                is_followup=False
                            )
                            st.session_state.simplified_content = simplified

    with col2:
        st.header("✨ Results & Chat")

        if st.session_state.simplified_content:
            # Display simplified content
            with st.container():
                st.markdown("### 📝 Simplified Content")
                st.markdown(st.session_state.simplified_content)

            # Show translated content if available
            if st.session_state.translated_content and available_languages[translate_to]:
                st.markdown("---")
                st.markdown("### 🌐 Translated Content")
                st.markdown(st.session_state.translated_content)

            # Interactive chat section
            st.markdown("---")
            st.markdown("### 💬 Refine with Chat")

            # User feedback input
            user_feedback = st.text_area(
                "Chat with AI to refine the content:",
                placeholder="e.g., 'Make it simpler', 'Add more examples', 'Focus on practical applications', 'Explain the technical terms better'...",
                height=100,
                key="user_feedback_input"
            )

            col_chat1, col_chat2, col_chat3 = st.columns(3)

            with col_chat1:
                if st.button("💬 Send Message", use_container_width=True):
                    if user_feedback:
                        with st.spinner("🤖 AI is thinking..."):
                            response = simplifier.simplify_content_with_chat(
                                st.session_state.original_content,
                                complexity_level,
                                focus_area,
                                model_choice,
                                user_feedback,
                                is_followup=True
                            )
                            st.session_state.simplified_content = response
                            st.rerun()

            with col_chat2:
                if st.button("🔄 Regenerate", use_container_width=True):
                    with st.spinner("🤖 Regenerating..."):
                        response = simplifier.simplify_content_with_chat(
                            st.session_state.original_content,
                            complexity_level,
                            focus_area,
                            model_choice,
                            "Please provide a different version of the simplification",
                            is_followup=True
                        )
                        st.session_state.simplified_content = response
                        st.rerun()

            with col_chat3:
                if st.button("🌐 Translate", use_container_width=True):
                    if available_languages[translate_to] and st.session_state.simplified_content:
                        with st.spinner("🌐 Translating..."):
                            translated = simplifier.translate_text(
                                st.session_state.simplified_content,
                                target_language=available_languages[translate_to],
                                model_choice=model_choice
                            )
                            st.session_state.translated_content = translated
                            st.rerun()

            # Chat history display
            if st.session_state.chat_history:
                with st.expander("📜 Conversation History"):
                    for entry in st.session_state.chat_history[-6:]:
                        role_icon = "👤" if entry['type'] == 'user' else "🤖"
                        timestamp = datetime.fromisoformat(entry['timestamp']).strftime("%H:%M:%S")
                        st.markdown(f"**{role_icon} {timestamp}** - {entry['content'][:200]}...")

            # Download options
            st.markdown("---")
            st.markdown("### 📥 Download Options")

            col_dl1, col_dl2, col_dl3 = st.columns(3)

            with col_dl1:
                st.download_button(
                    label="📄 Simplified",
                    data=st.session_state.simplified_content,
                    file_name="simplified_content.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            with col_dl2:
                comparison = f"""ORIGINAL CONTENT:
{'-'*50}
{st.session_state.original_content[:2000]}...

SIMPLIFIED CONTENT:
{'-'*50}
{st.session_state.simplified_content}

CONVERSATION HISTORY:
{'-'*50}
{json.dumps(st.session_state.chat_history[-5:], indent=2)}
"""
                st.download_button(
                    label="📋 Full Report",
                    data=comparison,
                    file_name="content_analysis_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            with col_dl3:
                if st.session_state.translated_content:
                    st.download_button(
                        label="🌐 Translated",
                        data=st.session_state.translated_content,
                        file_name=f"translated_{translate_to.lower().replace(' ', '_')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

        else:
            st.info("👆 Upload a file and click 'Simplify Content' to begin")

            # Feature showcase
            st.markdown("""
            ### 🎯 Enhanced Features:

            **🤖 AI Conversation:**
            - Chat with AI to refine results
            - Memory of previous interactions
            - Context-aware responses

            **📁 File Support:**
            - PDFs (text-based & scanned)
            - Images (PNG, JPG, etc.) with OCR
            - Automatic text extraction

            **🧠 Smart Memory:**
            - Remembers conversation context
            - Learns from your feedback
            - Maintains session history

            **💬 Interactive Refinement:**
            - Ask for specific changes
            - Request different examples
            - Adjust complexity on-the-fly
            """)

    # Footer
    footer="""<style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: black;
        text-align: center;
        z-index: 999;
    }
    </style>
    <div class="footer">
    <p>Enhanced AI Simplifier with Memory & Chat 🤖💭</p>
    </div>
    """
    # st.markdown(footer, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
