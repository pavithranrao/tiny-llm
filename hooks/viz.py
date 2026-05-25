"""MkDocs hook: injects raw HTML files into pages, bypassing Markdown sanitization."""
import re
from pathlib import Path


def on_page_markdown(markdown, *, page, config, **kwargs):
    """Replace {{ viz:path/to/file.html }} with the full HTML file content."""
    project_dir = Path(config["config_file_path"]).parent
    docs_dir = Path(config["docs_dir"])

    def replace(match):
        rel_path = match.group(1)
        source = project_dir / rel_path
        if not source.exists():
            source = docs_dir / rel_path
        if not source.exists():
            return f"<!-- MISSING: {rel_path} -->"
        return source.read_text()

    return re.sub(r'\{\{\s*viz:\s*(.+?)\s*\}\}', replace, markdown)


def on_post_page(output, *, page, config, **kwargs):
    """Fix injected HTML: strip document wrapper, move styles to head, keep scripts in body."""
    head_close = output.find("</head>")
    body_close = output.rfind("</body>")
    if head_close == -1 or body_close == -1:
        return output

    body = output[head_close:body_close]

    # 1) Remove any nested <!DOCTYPE>, <html>, <head>, </head>, <body>, </body> tags
    body = re.sub(r'<!DOCTYPE[^>]*>', '', body, flags=re.IGNORECASE)
    body = re.sub(r'<html[^>]*>', '', body, flags=re.IGNORECASE)
    body = re.sub(r'</html>', '', body, flags=re.IGNORECASE)
    body = re.sub(r'<head[^>]*>', '', body, flags=re.IGNORECASE)
    body = re.sub(r'</head>', '', body, flags=re.IGNORECASE)
    body = re.sub(r'<body[^>]*>', '', body, flags=re.IGNORECASE)
    body = re.sub(r'</body>', '', body, flags=re.IGNORECASE)
    # Remove <title> from nested doc
    body = re.sub(r'<title>[^<]*</title>', '', body, flags=re.IGNORECASE)
    # Remove charset/viewport meta tags from nested doc
    body = re.sub(r'<meta[^>]*charset[^>]*>', '', body, flags=re.IGNORECASE)
    body = re.sub(r'<meta[^>]*viewport[^>]*>', '', body, flags=re.IGNORECASE)

    # 2) Extract <style> blocks → inject into <head>
    styles = []
    def extract_style(m):
        styles.append(m.group(0))
        return ""
    body = re.sub(r'<style[^>]*>.*?</style>', extract_style, body, flags=re.DOTALL)

    if not styles:
        return output

    style_inject = "\n".join(styles)
    return output[:head_close] + style_inject + "\n</head>" + body + output[body_close:]
