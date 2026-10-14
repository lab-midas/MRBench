# CMRBench: interactive reconstruction results

This first publication contains the interactive dashboard and 1,755 audited,
anonymized subject-level results for acceleration and k-t sampling experiments.
Qualitative evaluation and all other result sections are empty placeholders.
Reconstruction-framework source will be added separately; the static framework
guides are labeled as a documentation preview.

## Preview

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory website
```

Open http://localhost:8000 on the computer running that command. For a remote
Linux server, forward its port from your local computer with
`ssh -N -L 8000:127.0.0.1:8000 USER@SERVER` and open the same local URL.

## Publish with GitHub Pages

Create `midas-tum/CMRBench` with main as the default branch, push this directory's
contents, then select Settings > Pages > Source > GitHub Actions. The included
workflow validates the data and deploys only `website/`.
The default project URL after successful deployment is
https://midas-tum.github.io/CMRBench/ . It is not a live deployment until the
workflow completes successfully.

No patient images, raw arrays, model weights, private ID map or reconstruction
sampling implementation is included. Author/affiliation and first-party license
details remain pending. The main metric protocol is described in the dashboard.
