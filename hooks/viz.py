"""MkDocs hook: injects raw HTML files into pages, bypassing Markdown sanitization."""
import re
from pathlib import Path


def _extract_body(html: str) -> str:
    """Strip document wrapper — extract only <style>, <script>, and <body> content."""
    # Collect <style> blocks from <head>
    styles = re.findall(r'<style[^>]*>.*?</style>', html, re.DOTALL)
    # Collect <script> blocks
    scripts = re.findall(r'<script[^>]*>.*?</script>', html, re.DOTALL)
    # Extract <body> inner content
    body_match = re.search(r'<body[^>]*>(.*)</body>', html, re.DOTALL)
    body_content = body_match.group(1) if body_match else html
    return "\n".join(styles + [body_content] + scripts)


def on_page_markdown(markdown, *, page, config, **kwargs):
    """Replace {{ viz:path/to/file.html }} placeholders with raw HTML content."""
    project_dir = Path(config["config_file_path"]).parent
    docs_dir = Path(config["docs_dir"])

    def replace(match):
        rel_path = match.group(1)
        source = project_dir / rel_path
        if not source.exists():
            source = docs_dir / rel_path
        if not source.exists():
            return f"<!-- MISSING: {rel_path} -->"
        return _extract_body(source.read_text())

    return re.sub(r'\{\{\s*viz:\s*(.+?)\s*\}\}', replace, markdown)


def on_post_page(output, *, page, config, **kwargs):
    """Move all <style> and <script> from body content into <head>."""
    head_close = output.find("</head>")
    if head_close == -1:
        return output

    # Collect <style> blocks from the article content
    styles = []
    def extract_style(m):
        styles.append(m.group(0))
        return ""

    body_content = output[head_close:]
    body_content = re.sub(r"<style[^>]*>.*?</style>", extract_style, body_content, flags=re.DOTALL)

    # Collect <script> blocks from the article content (except mkdocs/theme scripts)
    scripts = []
    def extract_script(m):
        scripts.append(m.group(0))
        return ""

    body_content = re.sub(r"<script[^>]*>.*?</script>", extract_script, body_content, flags=re.DOTALL)

    if not styles and not scripts:
        return output

    inject = "\n".join(styles + scripts)
    return output[:head_close] + inject + "\n</head>" + body_content
