The script is designed to run top to bottom in one go, but in practice you will want to run it in sections because the audio loading (Section 6) takes 1–3 hours and the model fitting (Section 9) takes another 1–2 hours.
The file = argument in each brm() call means models are cached — if you restart R and rerun, they load instantly from disk instead of refitting.

Recommended workflow:

Run Sections 1–5 first to verify data loads correctly
Run Section 6 and let audio loading complete overnight
Run Sections 7–9 to fit models
Run Sections 10–14 for results and figures
