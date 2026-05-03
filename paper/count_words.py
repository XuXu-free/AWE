import re
import os

contents_dir = 'C:/Users/admin/Desktop/sjtu/AWE/paper/contents'
total_chinese = 0
total_english_words = 0
files = []

for fname in os.listdir(contents_dir):
    if fname.endswith('.tex'):
        fpath = os.path.join(contents_dir, fname)
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Remove comments
        content = re.sub(r'%.*?\n', '\n', content)
        # Remove LaTeX commands
        content = re.sub(r'\\[a-zA-Z]+(\[.*?\])?(\{.*?\})?', ' ', content)
        # Remove math
        content = re.sub(r'\$.*?\$', ' ', content)
        content = re.sub(r'\\begin\{.*?\}.*?\\end\{.*?\}', ' ', content, flags=re.DOTALL)
        content = re.sub(r'\{.*?\}', ' ', content)

        chinese_chars = re.findall(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', content)
        chinese_count = len(chinese_chars)
        english_words = re.findall(r'[a-zA-Z]+', content)
        english_count = len(english_words)

        total_chinese += chinese_count
        total_english_words += english_count
        files.append((fname, chinese_count, english_count))

print('%-30s %8s %8s' % ('File', 'Chinese', 'English'))
print('-' * 50)
for fname, c, e in sorted(files):
    print('%-30s %8d %8d' % (fname, c, e))
print('-' * 50)
print('%-30s %8d %8d' % ('Total', total_chinese, total_english_words))
print('\nEstimated word count (Chinese chars + English words): %d' % (total_chinese + total_english_words))
