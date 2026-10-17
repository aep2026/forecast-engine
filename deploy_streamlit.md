# Deploying the scenario app (GitHub + Streamlit Community Cloud)

1. github.com -> New repository -> name `forecast-engine`, Public, no README -> Create.
2. "uploading an existing file" -> drag the CONTENTS of the unzipped forecast-engine folder (not the folder itself) -> Commit changes.
3. Check the repo shows app.py, requirements.txt, packages.txt, app_data/ and forecast_engine/ at the top level.
4. share.streamlit.io -> sign in with GitHub -> Create app -> Deploy a public app from GitHub.
5. Repository: <you>/forecast-engine, Branch: main, Main file path: app.py. Advanced settings -> Python 3.12. Deploy.
6. First start takes 3-5 minutes (installing LightGBM, statsmodels). Then copy the app URL for the submission.
Updating: change files on GitHub -> the app redeploys by itself.
