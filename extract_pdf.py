import os
from PyPDF2 import PdfReader, PdfWriter


def extract_pages(input_path, output_path, start_page, end_page):
    """提取pdf文件的一个切片"""
    reader = PdfReader(input_path)
    writer = PdfWriter()

    # 页码从0开始，所以我们需要减1
    for page_num in range(start_page - 1, end_page):
        if page_num < len(reader.pages):
            writer.add_page(reader.pages[page_num])

    # 将提取的页面写入新的PDF文件
    with open(output_path, 'wb') as output_file:
        writer.write(output_file)


def main():
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 设置输入和输出文件路径
    pdf_folder = os.path.join(current_dir, "PDF Files")
    input_pdf_path = os.path.join(pdf_folder, "Rough Sets.pdf")
    output_pdf_path = os.path.join(pdf_folder, "A Framework of Three-Way Cluster Analysis.pdf")
    start_page = 324  # 开始页码
    end_page = 336  # 结束页码
    print(input_pdf_path)
    print(output_pdf_path)
    extract_pages(input_pdf_path, output_pdf_path, start_page, end_page)


if __name__ == '__main__':
    main()
