import PyPDF2
import os

from utils import simple_progress_bar

def pdf_to_txt(pdf_path, output_txt):
    # Open the PDF file in read-binary mode
    with open(pdf_path, 'rb') as pdf_file:
        # Create a PdfReader object instead of PdfFileReader
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Initialize an empty string to store the text
        text = ''

        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

    # Write the extracted text to a text file
    with open(output_txt, 'w', encoding='utf-8') as txt_file:
        txt_file.write(text)

# def pdf_to_txt(pdf_path, txt_path):
#     with fitz.open(pdf_path) as pdf:
#         with open(txt_path, "w", encoding="utf-8") as txt_file:
#             for page in pdf:
#                 txt_file.write(page.get_text())


def convert_dir(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        total = len(files)
        for i, file in enumerate(files):
            simple_progress_bar(i, total, prefix="Checking pdf files for conversion to txt")

            if not file.endswith(".pdf"):
                continue
            pdf_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, input_dir)
            txt_subdir = os.path.join(output_dir, relative_path)
            os.makedirs(txt_subdir, exist_ok=True)

            txt_path = os.path.join(txt_subdir, f"{os.path.splitext(file)[0]}.txt")

            # Check if TXT file already exists
            if os.path.exists(txt_path):
                print(f"Skipping (already exists): {txt_path}")
                continue

            print(f"Converting: {pdf_path} -> {txt_path}")
            pdf_to_txt(pdf_path, txt_path)
