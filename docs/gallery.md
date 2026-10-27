# Gallery

An interactive map of the archive and a 3D replay of each robot: [open the gallery](site/index.html).

To build it yourself:

```bash
# Static site with an interactive map of the archive and a 3D replay of each robot
python tools/make_site.py            # all buildable elites; --max 12 for a small one
python -m http.server -d docs/site   # preview at http://localhost:8000
```

GitHub Pages can serve `docs/site` once the repository is public.
