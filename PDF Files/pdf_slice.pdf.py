from PyPDF2 import PdfReader, PdfWriter


def extract_pages(input_path, output_path, start_page, end_page):
    """
    提取PDF文件的一个切片

    :param input_path: 被切片的文件
    :param output_path: 输入文件的一个切片
    :param start_page: 切片开始页码
    :param end_page: 切片结束页码
    :return: 返回pdf文件的一个切片
    """
    reader = PdfReader(input_path)  # PdfReader对象用于读取现有的PDF文件
    writer = PdfWriter()  # PdfWriter对象用于创建新的PDF文件或修改现有的PDF

    # 将提取的页面添加到新的pdf文件中
    for page_num in range(start_page-1, end_page):
        if page_num < len(reader.pages):  # len(reader.pages)表示pdf文件的总页面数
            # 将选定的页面加入到writer对象的内部结构中，这一过程在内存中进行，还没有创建新的文件
            writer.add_page(reader.pages[page_num])

    # 将writer对象中添加的页面写入pdf文件中，这一过程在磁盘中进行
    with open(output_path, 'wb') as output_file:  # wb: 以二进制写入模式打开文件
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
