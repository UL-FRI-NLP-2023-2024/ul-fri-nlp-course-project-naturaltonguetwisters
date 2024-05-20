import fitz

def pdf_to_text_pymupdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text("text")
    
    return text

filename = 'Remarkably_Bright_Creatures'

pdf_path = f'{filename}.pdf'
extracted_text = pdf_to_text_pymupdf(pdf_path)

with open(f'{filename}.txt', 'w', encoding='utf-8') as f:
    f.write(extracted_text)
