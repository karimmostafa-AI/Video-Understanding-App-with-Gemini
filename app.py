import streamlit as st
import os
import base64
import tempfile
import google.generativeai as genai
import curl_cffi
import speech_recognition as sr
import cv2
import moviepy.editor as mp

# Set up Gemini API key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def extract_audio_and_transcribe(video_path, language="en-US"):
    """Extract audio from video and transcribe it using the specified language."""
    video = None
    temp_audio = None
    try:
        # Load the video
        video = mp.VideoFileClip(video_path)
        
        # Extract audio and save temporarily with higher fps for better quality
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        video.audio.write_audiofile(temp_audio.name, fps=16000)
        
        # Initialize recognizer
        recognizer = sr.Recognizer()
        
        # Transcribe audio using the provided language setting
        with sr.AudioFile(temp_audio.name) as source:
            # Adjust for ambient noise to help capture speech better
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.record(source)
            try:
                transcript = recognizer.recognize_google(audio, language=language)
            except sr.UnknownValueError:
                st.write("No recognizable speech found.")
                transcript = "No recognizable speech found."
            except sr.RequestError as e:
                st.write(f"Request Error: {e}")
                transcript = ""
        
        # Debug: عرض النص المُستخرج للتأكد من صحته
        st.write("Debug Transcript:", transcript)
        return transcript
    except Exception as e:
        st.error(f"Error in audio transcription: {str(e)}")
        return ""
    finally:
        # Clean up resources
        if video is not None:
            video.close()
        if temp_audio is not None:
            temp_audio.close()
            try:
                os.unlink(temp_audio.name)
            except Exception as cleanup_error:
                st.error(f"Cleanup error: {cleanup_error}")

def video_to_base64_frames(video_file_path):
    """Smart frame extraction with keyframe detection"""
    video = cv2.VideoCapture(video_file_path)
    base64_frames = []
    
    try:
        # Get video properties
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames = 12  # Optimal frame count
        
        # Extract keyframes at intervals
        for i in range(max_frames):
            target_frame = int((i / max_frames) * total_frames)
            video.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            success, frame = video.read()
            
            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _, buffer = cv2.imencode('.jpg', frame)
                base64_frames.append(base64.b64encode(buffer).decode('utf-8'))
    finally:
        video.release()
    return base64_frames

def get_video_description(base64_frames, transcript):
    """Uses Gemini to generate a video description based on frames and audio."""
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    prompt = """Analyze this video sequence following these steps:
1. Describe ONLY what you can see in the frames (people, objects, actions)
2. Note any text or captions visible in the video
3. Consider the audio transcript: {transcript}
4. Provide a clear, factual summary of what's happening in the video

Focus on being accurate and specific about what you observe."""
    
    content = [{
        "text": prompt.format(transcript=transcript)
    }]
    content.extend([{
        "inline_data": {
            "mime_type": "image/jpeg",
            "data": frame
        }
    } for frame in base64_frames[:8]])  # Using optimal frame count
    
    try:
        response = model.generate_content(content)
        return response.text
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return "Analysis unavailable"

def main():
    st.title("Video Understanding App (Gemini 2.0)")

    # Allow user to select the audio language
    language_option = st.selectbox(
        "Select audio language",
        options=["English (en-US)", "Arabic (ar-EG)"],
        index=0
    )
    lang_code = "en-US" if language_option.startswith("English") else "ar-EG"

    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        try:
            # Save the video to a temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            tfile.close()

            with st.spinner('Processing video...'):
                # Extract frames
                base64_frames = video_to_base64_frames(tfile.name)
                st.image(base64.b64decode(base64_frames[0]), caption="First frame of the video")

                # Get transcript using the selected language
                transcript = extract_audio_and_transcribe(tfile.name, language=lang_code)
                if transcript:
                    st.write("Transcript:", transcript)

                # Generate description using both video and audio
                description = get_video_description(base64_frames, transcript)
                st.write("Video Analysis:", description)
        finally:
            # Clean up temp file
            try:
                os.unlink(tfile.name)
            except Exception as unlink_error:
                st.error(f"Error deleting temp file: {unlink_error}")

if __name__ == "__main__":
    main()
