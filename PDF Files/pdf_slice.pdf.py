from PyPDF2 import PdfReader, PdfWriter


def extract_pages(input_path, output_path, start_page, end_page):
    """
    提取pdf文件的一个切片

    :param input_path: 被切片的文件
    :param output_path: 输入文件的一个切片
    :param start_page: 切片开始页码
    :param end_page: 切片结束页码
    :return:
    """
    reader = PdfReader(input_path)
    writer = PdfWriter()
    print(len(reader.pages))

    # 页码从0开始，所以我们需要减1
    for page_num in range(start_page - 1, end_page):
        # len(reader.pages)表示pdf文件的总页面数
        if page_num < len(reader.pages):
            writer.add_page(reader.pages[page_num])

    # 将提取的页面写入新的PDF文件
    with open(output_path, 'wb') as output_file:
        writer.write(output_file)


def main():
    # 设置输入和输出文件路径
    input_pdf_path = "Rough Sets.pdf"
    output_pdf_path = "A Framework of Three-Way Cluster Analysis.pdf"
    start_page = 324  # 开始页码
    end_page = 336  # 结束页码
    extract_pages(input_pdf_path, output_pdf_path, start_page, end_page)


if __name__ == '__main__':
    main()
