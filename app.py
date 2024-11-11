import streamlit as st
import moviepy.editor as me
import librosa
import numpy as np
import gc
import torch 
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from nltk.tokenize import sent_tokenize
import re
import pandas as pd
from sentence_transformers import util, SentenceTransformer
from tqdm import tqdm
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from dotenv import load_dotenv
import cv2
import os
import pytubefix
from pytubefix import YouTube
from pytubefix.cli import on_progress
import requests
#ocr
import pytesseract
import os
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


#video
from llama_index.core.schema import ImageNode
import os
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

from huggingface_hub import InferenceClient



if 'image_processed' not in st.session_state:
    st.session_state.image_processed=False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_input' not in st.session_state:
    st.session_state.user_input=''
if 'embedded' not in st.session_state:
    st.session_state.embedded=False
if 'llm_model' not in st.session_state:
    st.session_state.llm_model=None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer=None
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text=None
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model=None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings=None
if 'chunks' not in st.session_state:
    st.session_state.chunks=None
if 'model_whisper' not in st.session_state:
    st.session_state.model_whisper=None
if 'query_input' not in st.session_state:
    st.session_state.query_input=None
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'output_folder' not in st.session_state:
    st.session_state.output_folder='temp_video_images'
if 'retriever_engine' not in st.session_state:
    st.session_state.retriever_engine=None

def extract_text_from_frame(frame):
    # Convert the frame to grayscale for better OCR accuracy
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply OCR to the grayscale frame
    text = pytesseract.image_to_string(gray_frame)
    
    return text



def process_video_for_ocr(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    process_interval = 15

    with open('extracted_text.txt', 'w') as file:
        while True:
            ret, frame = cap.read()
            if not ret:
                # print("Something went wrong or end of video.")
                break
            
            if frame_count % process_interval == 0:
                # print(f'Processing Frame {frame_count}...')
                text = extract_text_from_frame(frame)
                if text.strip():
                    file.write(f"Frame {frame_count}:\n{text}\n\n")
            
            frame_count += 1
            
    cap.release()

def clean_text_from_file():
    cleaned_lines = []
    
    with open('extracted_text.txt', 'r') as infile:
        lines = infile.readlines()

    for line in lines:
        cleaned_line = ' '.join(line.split()).strip()
        if cleaned_line:  # Only add non-empty lines
            cleaned_lines.append(cleaned_line)

    # Join the cleaned lines into a single string if needed
    cleaned_text = '\n'.join(cleaned_lines)
    print("OCE process completed")
    return cleaned_text





def reg_video():
    image_store=LanceDBVectorStore(uri="lancedb",table_name="image_collection")
    storage_context=StorageContext.from_defaults(image_store=image_store)
    documents=SimpleDirectoryReader(st.session_state.output_folder).load_data()
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core import Settings

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
    index = MultiModalVectorStoreIndex.from_documents(documents,storage_context=storage_context)
    retriever_engine=index.as_retriever( image_similarity_top_k=1)
    st.session_state.retriever_engine=retriever_engine
def video_retrieve(query_str):
    retriever_engine=st.session_state.retriever_engine
    retrieval_results = retriever_engine.retrieve(query_str)

    retrieved_image = []
    #retrieved_text = []
    for res_node in retrieval_results:
        if isinstance(res_node.node, ImageNode):
            retrieved_image.append(res_node.node.metadata["file_path"])
        #else:
            #display_source_node(res_node, source_length=200)
            #retrieved_text.append(res_node.text)

    return retrieved_image

def clear_gpu_memory(model):
    # del st.session_state.model_whisper
    # gc.collect()
    # torch.cuda.empty_cache()
    st.session_state.model_whisper=None
    # model.to('cpu')
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("Model deleted and GPU memory cleared.")

# def get_file():
#     def audio_file(file):
#         if uploaded_file_name[-4:] in ['.mp4', '.mkv', '.mov']:
#             video = me.VideoFileClip(uploaded_file_name)
#             audio = video.audio
#             audio.write_audiofile('temp.mp3')
#             file_path = 'temp.mp3'
#         else:
#             file_path = file
#         audio_data, sampling_rate = librosa.load(file_path, sr=16000)
#         audio_info = {'path': file_path, 'array': audio_data, 'sampling_rate': sampling_rate}
#         return audio_info
#     def video_to_image(file):
#         output_folder="temp_video_images"
#         if os.path.exists(output_folder):
#             # Iterate over all files in the directory
#             for filename in os.listdir(output_folder):
#                 file_path = os.path.join(output_folder, filename)
#                 # Check if it is a file
#                 if os.path.isfile(file_path):
#                     os.remove(file_path)  # Delete the file
#                     # print(f"{file_path} has been deleted.")
        
#         cap = cv2.VideoCapture(file)

#         # Get the video frame rate
#         fps = cap.get(cv2.CAP_PROP_FPS)

#         # Calculate the interval to capture frames (0.2 FPS means every 5 seconds)
#         frame_interval = int(fps * 2)

#         frame_count = 0
#         saved_frame_count = 0

#         # Loop through frames in the video
#         while cap.isOpened():
#             ret, frame = cap.read()
            
#             if not ret:
#                 break
            
#             # Save the frame every 5 seconds
#             if frame_count % frame_interval == 0:
#                 frame_name = f"{output_folder}/frame_{saved_frame_count:04d}.jpg"
#                 cv2.imwrite(frame_name, frame)
#                 saved_frame_count += 1
            
#             frame_count += 1

#         # Release the video capture
#         cap.release()
#         print(f"Frames saved in '{output_folder}'")
#         reg_video()
#     uploaded_file = st.file_uploader("Upload an audio or video file", type=['mp3', 'wav', 'mp4', 'mkv', 'mov'])

#     if uploaded_file is not None:
#         uploaded_file_name = uploaded_file.name
#         uploaded_file_name = "temp" + uploaded_file_name[-4:]
#         with open(uploaded_file_name, 'wb') as f:
#             f.write(uploaded_file.getbuffer())

#         audio_file_info = audio_file(uploaded_file_name)
#         video_to_image(uploaded_file_name)
#         print(video_retrieve("who is the person in this video"))
#         # process_video_for_ocr(uploaded_file_name)
#         # cleaned_text = clean_text_from_file()
#         # print(cleaned_text)
#         get_text(audio_file_info)

def get_text(audio_path):
    mode_creation()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id_whisper = "openai/whisper-large-v3"

    model_whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id_whisper, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model_whisper.to(device)

    processor_whisper = AutoProcessor.from_pretrained(model_id_whisper)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_whisper,
        tokenizer=processor_whisper.tokenizer,
        feature_extractor=processor_whisper.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )
    result = pipe(audio_path, generate_kwargs={"task": "translate"})
    st.session_state.model_whisper = model_whisper
    st.session_state.extracted_text = result['text']
    st.session_state['messages'].append({"role": "bot", "content": f'The content extracted from the audio file: {result["text"]}'})
    clear_gpu_memory(model_whisper)
    summarizer(result["text"])

def mode_creation():
    load_dotenv()
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    llm = Ollama(model="llama3")
    output_parser = StrOutputParser()
    st.session_state.llm_model = llm
    st.session_state.tokenizer = output_parser

def prompt_formatter_summ():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Based on the following text, please give the summary of the text. Don't return the thinking, only return the answer. Make sure your answers are as explanatory as possible.\nNow use the following text to summarize the text: {text}\nAnswer:"),
            ("user", "Provide the summary of the given audio from a video.")
        ]
    )
    return prompt

def summarizer(text, temperature=0.7, max_new_tokens=512, format_answer_text=True, return_answer_only=True):
    prompt = prompt_formatter_summ()
    llm_model=st.session_state.llm_model
    tokenizer=st.session_state.tokenizer
    chain = prompt | llm_model |tokenizer 
    print("Stated")
    output_text = chain.invoke({"text": text})
    print("Text generated")
    st.session_state['messages'].append({"role": "bot", "content": f'The summary of the file: {output_text}'})
    st.session_state.image_processed = True
    embedding()

def embedding():
    text = st.session_state.extracted_text
    sentences_of_text = sent_tokenize(text)
    num_sentence_chunk_size = 10

    def split_list(input_list, slice_size):
        return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

    sentence_chunks = split_list(sentences_of_text, num_sentence_chunk_size)
    chunks = []
    for sentence_chunk in sentence_chunks:
        chunk_dict = {}
        joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
        joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)
        chunk_dict["sentence_chunk"] = joined_sentence_chunk
        chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
        chunk_dict["chunk_word_count"] = len(joined_sentence_chunk.split(" "))
        chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4
        chunks.append(chunk_dict)

    df = pd.DataFrame(chunks)
    min_token_length = 30
    chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")

    embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cuda")
    embedding_model.to("cuda")

    for chunk in tqdm(chunks_over_min_token_len):
        chunk["embedding"] = embedding_model.encode(chunk["sentence_chunk"])

    text_chunks_and_embeddings_df = pd.DataFrame(chunks_over_min_token_len)
    text_chunks_and_embeddings_df.to_csv("text_chunks_and_embeddings_df.csv", index=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_chunks_and_embedding_df = pd.read_csv("text_chunks_and_embeddings_df.csv")
    text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    chunks = text_chunks_and_embedding_df.to_dict(orient="records")
    embeddings = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32).to(device)

    st.session_state.embedding_model = embedding_model
    st.session_state.embeddings = embeddings
    st.session_state.chunks = chunks

def prompt_formatter_rag():
    prompt = ChatPromptTemplate.from_messages(
       [
    ("system", """You are a video analyst. Your task is to perform video analysis based on the provided contents. The content retrieved from the image is labeled as 'Image Content,' and the content retrieved from the audio is labeled as 'Audio Content.' Answer the query based solely on these contents without adding any personal interpretation or additional thoughts.\n\nImage Content: {image_content}\nAudio Content: {audio_content}"""),
    ("user", "Query: {query}")
]

    )

    return prompt

def retriever_score(query):
    embedding_model = st.session_state.embedding_model
    embeddings = st.session_state.embeddings
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    dot_scores = util.dot_score(a=query_embedding, b=embeddings)[0]
    scores, indices = torch.topk(input=dot_scores, k=1)
    return scores, indices

def rag_answers(query,image_content, temperature=0.8, max_new_tokens=512, format_answer_text=True):
    chunks = st.session_state.chunks
    scores, indices = retriever_score(query)
    context_items = [chunks[i] for i in indices]
    for i, item in enumerate(context_items):
        item["score"] = scores[i].cpu()

    prompt = prompt_formatter_rag()
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])
    chain = prompt | st.session_state.llm_model | st.session_state.tokenizer
    print(image_content)
    output_text = chain.invoke({'image_content':image_content,"audio_content":context,"query":query})
    st.session_state['messages'].append({"role": "bot", "content": output_text})

# def main():
#     st.title("Video Understanding App")
    
#     if not st.session_state.image_processed:
#         get_file()
  
#     st.write("Chat with the file")

#     with st.form(key='chat_form', clear_on_submit=True):
#         user_input = st.text_input("You:", "")
#         submit_button = st.form_submit_button(label='Send')

#     if submit_button and user_input:
#         st.session_state['messages'].append({"role": "user", "content": user_input})
#         if user_input.lower()=='exit':
#             st.session_state.image_processed=None
#             st.session_state['messages']=[]
#             os.remove("temp.mp3")
#             os.remove('text_chunks_and_embeddings_df.csv')
#             st.stop()   
            
#         response = rag_answers(user_input)
#         video_file=video_retrieve(user_input)
#         st.session_state['messages'].append({"role": "bot", "content": response})
    
#     for message in st.session_state['messages'][::-1]: 
#         if message["role"] == "user":
#             st.write(f"You: {message['content']}")
#         else:
#             st.write(f"Bot: {message['content']}")


def audio_file(file):
    print(file)
    # if file[-4:] in ['.mp4', '.mkv', '.mov']:
    video = me.VideoFileClip(file)
    audio = video.audio
    audio.write_audiofile('temp.mp3')
    file_path = 'temp.mp3'
    # else:
    #     file_path = file
    audio_data, sampling_rate = librosa.load(file_path, sr=16000)
    audio_info = {'path': file_path, 'array': audio_data, 'sampling_rate': sampling_rate}
    return audio_info

def video_to_image(file):
    output_folder = "temp_video_images"
    
    # Clean up previous images if any
    if os.path.exists(output_folder):
        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    cap = cv2.VideoCapture(file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * 2)
    frame_count = 0
    saved_frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_name = f"{output_folder}/frame_{saved_frame_count:04d}.jpg"
            cv2.imwrite(frame_name, frame)
            saved_frame_count += 1
        frame_count += 1

    cap.release()
    print(f"Frames saved in '{output_folder}'")
    reg_video()

def upload_video_file(file_path):
    url = 'http://117.193.240.172:47/upload'

    try :

        # Open the image in binary mode and send the request
        with open(file_path, 'rb') as img:
            files = {'file': img}
            response = requests.post(url, files=files)

        print(response.json()) 
    except :
        print("Cant connect to the server url :",url)
    

def image_process_api_llama(query):
    client = InferenceClient(api_key="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")
    query=query
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": """You are a video analyst. Your task is to analyze the provided image retrieved from the RAG system and answer the query accurately. 
                Provide only relevant details for the question, without additional commentary.
                \n\n

                
                Query: {query}"""
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "http://117.193.240.172:47/image.jpg"
                }
            }
        ]
    }]
    
    try :
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-11B-Vision-Instruct", 
            messages=messages
        )
        print(response['choices'][0]['message']['content'])
        return(response['choices'][0]['message']['content'])
    except :
        print("Error can get the responce from the huggingface.")
        return None

def download_youtube_video(url):
    yt = YouTube(url, on_progress_callback=on_progress)
    print(f"Downloading: {yt.title}")

    # Get the highest resolution available
    ys = yt.streams.get_highest_resolution()
    output_path = ""
    # os.makedirs(output_path, exist_ok=True)
    
    # Download the video and return its path
    ys.download(output_path=output_path,filename="temp.mp4")
    video_path = os.path.join(output_path, "temp.mp4")
    print(f"Downloaded video saved to: {video_path}")
    return video_path

def get_file():
    uploaded_file = st.file_uploader("Upload an audio or video file, or paste a YouTube URL", type=['mp3', 'wav', 'mp4', 'mkv', 'mov'])
    youtube_url = st.text_input("Or provide a YouTube URL (if applicable)", "")

    if uploaded_file is not None:
        uploaded_file_name = uploaded_file.name
        uploaded_file_name = "temp" + uploaded_file_name[-4:]
        with open(uploaded_file_name, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        audio_file_info = audio_file(uploaded_file_name)
        video_to_image(uploaded_file_name)
        # print(video_retrieve("who is the person in this video"))
        get_text(audio_file_info)
    
    elif youtube_url:
        video_path = download_youtube_video(youtube_url)
        print("Video downloaded")
        audio_file_info = audio_file(video_path)
        video_to_image(video_path)
        get_text(audio_file_info)

def main():
    st.title("Cross-Modal LMM for Real-Time Video and Audio  Analysis")
    
    if not st.session_state.image_processed:
        get_file()
  
    st.write("Chat with the file")

    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_input("You:", "")
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        st.session_state['messages'].append({"role": "user", "content": user_input})
        if user_input.lower() == 'exit':
            print("Processing exit")
            st.session_state.image_processed = None
            st.session_state['messages'] = []
            os.remove("temp.mp3")
            os.remove('text_chunks_and_embeddings_df.csv')
            clear_gpu_memory(st.session_state.llm_model)
            torch.cuda.empty_cache() 
            import gc
            gc.collect()
            st.experimental_rerun()  

        video_file = video_retrieve(user_input)
        upload_video_file(video_file[0])
        print(video_file[0])
        video_output=image_process_api_llama(user_input)
        print("Image processed completed")
        response = rag_answers(user_input,video_output)
        st.session_state['messages'].append({"role": "bot", "content": response})
    
    for message in st.session_state['messages'][::-1]: 
        if message["role"] == "user":
            st.write(f"You: {message['content']}")
        else:
            st.write(f"Bot: {message['content']}")





if __name__ == "__main__":
    main()