# xuewyang.github.io

Personal site of Xuewen Yang — built with [Hugo](https://gohugo.io/) and the
[PaperMod](https://github.com/adityatelange/hugo-PaperMod) theme.

Deployed at **https://xuewyang.github.io/** via GitHub Actions on every push to `main`.

## Local development

```bash
# Install Hugo extended (>= 0.128)
brew install hugo            # macOS
# or: sudo snap install hugo  # Ubuntu

# Run the dev server
hugo server -D
# then open http://localhost:1313
```

## Repository layout

```
.
├── hugo.yaml               # Site config
├── content/
│   ├── posts/              # Blog posts (Markdown)
│   ├── publications.md     # Publications page
│   ├── projects.md         # Projects page
│   ├── search.md           # PaperMod search page
│   └── archives.md         # Archive page
├── themes/PaperMod/        # Vendored PaperMod theme (checked in)
├── static/                 # Images, favicons, CVs, anything served as-is
└── .github/workflows/
    └── hugo.yaml           # Build + deploy to GitHub Pages
```

## Writing a new blog post

```bash
hugo new posts/my-new-post.md
```

This creates a Markdown file in `content/posts/` with the right front matter.
Edit the body, commit, and push to `main` — the GitHub Action deploys the site
within a minute or two.

To mark a post as a draft (hidden from the live site), add `draft: true` to the
front matter.

## Updating the PaperMod theme

The theme is checked in as a plain directory (no submodule). To update it:

```bash
rm -rf themes/PaperMod
git clone --depth 1 https://github.com/adityatelange/hugo-PaperMod.git themes/PaperMod
rm -rf themes/PaperMod/.git
git add themes/PaperMod
git commit -m "Update PaperMod theme"
```

## First-time GitHub Pages setup

1. Push this repo to `github.com/xuewyang/xuewyang.github.io`.
2. In the repo, go to **Settings → Pages**.
3. Under **Build and deployment → Source**, select **GitHub Actions**.
4. The next push to `main` will deploy the site.

## Customizing

- **Profile photo** — drop a 200×200 image at `static/profile.png`, then add
  `profileMode` to `params` in `hugo.yaml`.
- **Favicon** — replace `static/favicon.ico` (the one here is a placeholder).
- **Custom domain** — add a `static/CNAME` file containing your domain, and
  update `baseURL` in `hugo.yaml`.
- **Custom CSS** — put overrides at `assets/css/extended/custom.css`.
