# 提取pdf文件的一个切片

from PyPDF2 import PdfReader, PdfWriter


def extract_pages(input_path, output_path, start_page, end_page):
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
    # 使用示例
    input_pdf = "input.pdf"  # 输入PDF文件路径
    output_pdf = "output.pdf"  # 输出PDF文件路径
    start_page = 33  # 开始页码
    end_page = 65  # 结束页码

    extract_pages(input_pdf, output_pdf, start_page, end_page)
    print(f"已从第 {start_page} 页到第 {end_page} 页提取并保存到 {output_pdf}")


if __name__ == '__main__':
    main()
