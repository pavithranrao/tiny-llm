"""MkDocs hook: injects raw HTML files into pages, bypassing Markdown sanitization."""
import re
from pathlib import Path


def on_page_markdown(markdown, *, page, config, **kwargs):
    """Replace {{ viz:path/to/file.html }} placeholders with raw HTML content."""
    docs_dir = Path(config["docs_dir"])
    project_dir = Path(config["config_file_path"]).parent

    def replace(match):
        rel_path = match.group(1)
        # Resolve relative to project root (for viz/ files)
        source = project_dir / rel_path
        if source.exists():
            return source.read_text()
        # Fallback: resolve relative to docs/
        source = docs_dir / rel_path
        if source.exists():
            return source.read_text()
        return f"<!-- MISSING: {rel_path} -->"

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
