The Kepler Object of Interest (KOI) catalog utilizes an automated vetting tool called the **Robovetter** to disposition Threshold Crossing Events (TCEs) into Planet Candidates (PCs) or False Positives (FPs). The Robovetter employs a variety of quantitative **metrics** and tests, drawing on two different detrended light curves: the Data Validation (DV) detrending and the Alternative (ALT) detrending.

The metrics primarily serve to test if a TCE falls into one of four major False Positive (FP) categories: Not-Transit-Like (NT), Stellar Eclipse (SS), Centroid Offset (CO), or Ephemeris Match Indicates Contamination (EC).

Below is an explanation of the core metrics and tests used by the Robovetter, organized by the FP category they primarily address, along with essential overall metrics:

---

### Core Catalog Characterization Metrics

While not used internally by the Robovetter logic to disposition individual TCEs, these metrics are crucial for characterizing the performance of the catalog:

*   **Catalog Reliability ($\mathbf{R}$):** Defined as the ratio of the number of PCs that are truly exoplanets ($T_{PC_{obs}}$) to the total number of observed PCs ($N_{PC_{obs}}$). This metric gauges the fraction of candidates that are not due to instrumental or stellar noise.
*   **Robovetter Completeness ($\mathbf{C}$):** The fraction of injected transits detected by the Kepler Pipeline that are subsequently passed by the Robovetter as PCs. It measures what fraction of true planets are missing from the final catalog.
*   **Robovetter Effectiveness ($\mathbf{E}$):** The fraction of simulated False Alarms (using inverted or scrambled data, $N_{sim}$) that are correctly identified by the Robovetter as FPs. Effectiveness is used to estimate catalog reliability.
*   **Multiple Event Statistic (MES):** A statistic measuring the combined significance of all observed transits in the detrended, whitened light curve, assuming a linear ephemeris.
*   **Disposition Score:** A value between 0 and 1 that indicates the Robovetter's confidence in a disposition, calculated using a Monte Carlo routine. Higher values indicate more confidence that a TCE is a PC.

---

### Not-Transit-Like (NT) Metrics

These metrics test whether the TCE light curve is physically consistent with a transiting or eclipsing system, primarily rejecting instrumental artifacts, statistical fluctuations, poor detrending, or stellar variability.

| Metric Name | Purpose and Mechanism | Citation |
| :--- | :--- | :--- |
| **LPP Metric (LPP\_DV/LPP\_ALT)** | Measures how similar the TCE's folded light curve shape is to known transits using Locality Preserving Projections (LPP). A high LPP value indicates the signal is not transit shaped. | |
| **SWEET NTL (SWEET\_NTL)** | Sine Wave Event Evaluation Test. Checks the PDC data for a strong sinusoidal signal at the TCE's period, suggesting stellar variability or a quasi-sinusoidal FP. Fails if the signal-to-noise ratio (S/N) of the sinusoidal fit is greater than 50, and its amplitude is greater than the TCE transit depth (for periods less than 5.0 days). | |
| **TCE Chases (ALL\_TRANS\_CHASES)** | Assesses the median detection strength of individual transits relative to nearby signals over time. Used only for TCEs with five or fewer transits; failure indicates individual events are not uniquely significant. | |
| **Model-Shift 1 (MS1)** | Determines if the primary event is statistically significant compared to the systematic noise level (red noise) in the light curve. Fails if the primary transit is not significant compared to the red noise. | |
| **Model-Shift 2 (MS2)** | Determines if there is a statistically significant negative event (tertiary event) in the phased light curve, suggesting the primary event is not unique. | |
| **Model-Shift 3 (MS3)** | Determines if there is a statistically significant positive flux event (brightening) in the phased light curve, suggesting the primary event is noise. | |
| **Max SES to MES (INCONSISTENT\_TRANS)** | Checks if the ratio of the maximum Single Event Statistic ($SES_{Max}$) to the Multiple Event Statistic (MES) is greater than 0.8 (for periods > 90 days), indicating that the detection significance is dominated by a single transit event, typical of long-period FPs. | |
| **Same Period (SAME\_NTL\_PERIOD/RESIDUAL\_TCE)** | Fails a TCE if its period matches a previous TCE already designated as a Not-Transit-Like FP (SAME\_NTL\_PERIOD) or if it matches a previous transit-like TCE and is separated in phase by less than 2.5 transit durations, suggesting it is a residual artifact (RESIDUAL\_TCE). | |
| **Gapped Transits (TRANS\_GAPPED)** | Fails a TCE if the fraction of individual transit events that actually contain data is too low (≤ 0.5). | |
| **Individual Transit Vetting (INDIV\_TRANS\_)** | A system of sub-metrics that analyze individual transit events. A TCE fails if fewer than three "good" events remain, or if the recomputed MES using only good events drops below 7.1. | |
| **Rubble (INDIV\_TRANS\_RUBBLE)** | Checks for missing data in the individual transit event; fails if the fraction of available cadences is below 0.75. | |
| **Marshall (INDIV\_TRANS\_MARSHALL)** | Uses a Gaussian process approach to model the out-of-transit continuum and compares the transit event to transient models (like sudden pixel sensitivity dropouts, SPSD). Fails if a non-transit model is statistically preferred over the transit model. | |
| **Chases (INDIV\_TRANS\_CHASES)** | Quantifies how uniquely significant an individual SES peak is relative to neighboring data in the SES time series. Fails if a feature of comparable strength is too close to the transit event. | |
| **Skye (INDIV\_TRANS\_SKYE)** | Looks for temporal clustering of individual transit events from TCEs in the same "skygroup" (region on the same CCD), indicative of systematic artifacts like rolling band noise. | |
| **Zuma (INDIV\_TRANS\_ZUMA)** | Designates an individual transit as "bad" if its SES is negative (SES < 0), indicating a flux increase instead of a decrease. | |
| **Tracker (INDIV\_TRANS\_TRACKER)** | Measures the difference between the TPS and DV linear ephemerides. If the difference is greater than 0.5 times the transit duration, it flags the transit as bad, suggesting the ephemeris wandered due to systematics. | |

---

### Stellar Eclipse (SS) Metrics

These metrics identify systems likely caused by stellar companions (eclipsing binaries) based on secondary eclipses, transit shape, or out-of-eclipse variability.

| Metric Name | Purpose and Mechanism | Citation |
| :--- | :--- | :--- |
| **Secondary TCE (HAS\_SEC\_TCE)** | Fails the TCE if a subsequent TCE on the same star has the same period but a significantly different epoch, indicating the subsequent TCE is a secondary eclipse. | |
| **MS Secondary (MOD\_SEC\_DV/ALT)** | Uses the model-shift test to detect a significant secondary eclipse in the light curve even if it didn't generate its own TCE. Fails if the secondary event is statistically significant compared to noise, the tertiary event, and the positive event. | |
| **Odd–Even (DEPTH\_ODDEVEN\_, MOD\_ODDEVEN\_)** | Compares the depths of odd- and even-numbered transits. If they are dissimilar, it suggests the system is an eclipsing binary detected at half its true orbital period. Two different methods (median depth vs. model-shift fit) are used to compute the significance difference ($\sigma_{OE}$). | |
| **SWEET EB (SWEET\_EB)** | Uses SWEET results on PDC data to detect out-of-eclipse variability caused by tidal deformation in short-period eclipsing binaries. Fails if a significant sinusoidal variation is present, but its amplitude is less than the TCE depth (for periods < 10 days). | |
| **V-Shape Metric (DEEP\_V\_SHAPED)** | Fails if the sum of the modeled radius ratio ($R_p/R\ast$) and the impact parameter ($b$) is greater than 1.04. This criterion identifies systems that are either too deep or grazing, which are highly likely to be eclipsing binaries. | |
| **Planet Occultation Override (PLANET\_OCCULT\_)** | Overrides an SS fail to PC if the detected secondary eclipse meets criteria indicating it might be a planetary occultation (e.g., low inferred geometric albedo, small planet radius). | |
| **Planet Half Period Override (PLANET\_PERIOD\_IS\_HALF\_)** | Overrides an SS fail to PC if the primary and secondary eclipses are statistically indistinguishable in width and depth and the secondary is located near phase 0.5, suggesting the TCE is a PC detected at twice the true orbital period. | |

---

### Centroid Offset (CO) Metrics

These metrics determine if the transit signal originates from a source other than the target star, indicating contamination from a background or nearby eclipsing binary.

| Metric Name | Purpose and Mechanism | Citation |
| :--- | :--- | :--- |
| **Resolved Offset (CENT\_RESOLVED\_OFFSET)** | Checks the difference image (flux during transit minus flux outside transit) to see if the brightest pixel is offset by more than 1.5 pixels from the target, suggesting contamination from a spatially resolved source. | |
| **Unresolved Offset (CENT\_UNRESOLVED\_OFFSET)** | Fits a pixel response function (PRF) model to the difference images to search for statistically significant shifts in the centroid during transit, suggesting contamination from an unresolved source. | |
| **Ghost Diagnostic (HALO\_GHOST)** | Measures the ratio of transit signal strength in the "halo" pixels (annulus surrounding the aperture) versus the "core" pixels. If the ratio is greater than 4.0, it suggests contamination from a diffuse ghost image or broad PRF wing of a nearby star. | |

---

### Ephemeris Match (EC) Metrics

This metric identifies contamination by comparing the TCE's period and epoch to those of known variable sources.

| Metric Name | Purpose and Mechanism | Citation |
| :--- | :--- | :--- |
| **Ephemeris Match (EPHEM\_MATCH)** | Fails the TCE if its period and epoch statistically match (SP < 5 and ST < 5) those of another object (another TCE, a known KOI, or a known eclipsing binary), provided the objects' locations are close enough to suggest contamination. | |

---

### Informational-Only Metrics

These tests provide warnings but do not automatically disposition a TCE as an FP, usually because they rely on highly uncertain parameters or the evidence is not definitive.

| Metric Name | Purpose and Mechanism | Citation |
| :--- | :--- | :--- |
| **Planet in Star (PLANET\_IN\_STAR)** | Flags cases where the DV fit results in an unphysical scenario where the planet's semimajor axis is smaller than the host star's radius. | |
| **Seasonal Depth Differences (SEASONAL\_DEPTH\_)** | Flags TCEs where the measured depth varies significantly (SDiff > 3.6) between the four Kepler "seasons" (based on spacecraft rotation), indicating significant light contamination. | |
| **Period Aliasing (PERIOD\_ALIAS\_)** | Flags TCEs where the secondary and tertiary event phases suggest that the pipeline detected the signal at an integer multiple (N:1) of the true orbital period. | |