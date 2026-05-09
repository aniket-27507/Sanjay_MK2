# AI Field Test Protocol — Balcony Validation

**Date:** TBD (tomorrow)
**Setup:** phone on balcony as drone-equivalent camera; laptop runs `police_full_v2.pt` AI; one operator + one subject.

---

## Goal

Validate that `police_full_v2.pt` detects the trained classes (person, weapon_person, vehicle, fire, explosive_device, crowd) at drone-equivalent altitude in real-world outdoor conditions. Generate evidence-of-validation suitable for the CM pitch.

## Hypothesis

At ~10m altitude (typical balcony / 3rd-floor window), with the phone tilted 30-45° down, the model should fire reliable alerts on `weapon_person` (toy rifle) and `person` classes. `explosive_device` (suspicious bag) and `fire` are stretch goals — out of training distribution at handheld scale, possibly fine at distance.

---

## Roles

| Role | Person | Job |
|------|--------|-----|
| **Operator** | Friend | Holds phone steady on balcony. Runs laptop. Watches dashboard. Classifies incidents (SAFE / THREAT / DISMISS). Records observations. |
| **Subject** | You | On the street below. Walks defined patterns with various props. Calls out to operator when entering / exiting frame. |

---

## Hardware checklist

- [ ] Android phone with **IP Webcam** app installed
- [ ] Laptop charged + power adapter
- [ ] Both devices on same WiFi (or laptop tethered to phone hotspot if no WiFi)
- [ ] `runs/detect/police_full_v2/weights/best.pt` on laptop
- [ ] Latest code from `origin/main` (commit `f7076176` or later)
- [ ] Phone tripod or fixed mount (handheld phone shake hurts detection)
- [ ] Props:
  - Toy rifle / realistic toy gun
  - 2-3 different bags: backpack, duffel, handbag (vary size + colour)
  - Lighter (for fire test, if safe at the location)
- [ ] Notebook + pen for ground-truth observations
- [ ] Stopwatch or phone timer
- [ ] Two-way comms (phones with intercom app, or just shouting distance)

---

## Pre-flight (do at the location, before subject walks)

15 minutes total.

1. **Mount phone** on balcony with mount/tripod, tilted 30-45° down so the street is visible in the lower 2/3 of the frame.
2. **IP Webcam → Start server.** Note the URL (e.g. `http://192.168.1.42:8080`).
3. **Laptop terminal 1:**
   ```powershell
   python scripts/demo_operator_workflow.py `
     --rgb-source http://192.168.1.42:8080/video `
     --rgb-model runs/detect/police_full_v2/weights/best.pt `
     --conf-threshold 0.20 `
     --alert-threshold 0.40 `
     --operator-timeout-sec 0 `
     --gcs-port 8765 `
     --audit-dir audit_runs/field_test_$(Get-Date -Format yyyyMMdd_HHmm)
   ```
4. **Laptop terminal 2:**
   ```powershell
   npm --prefix gcs-dashboard run dev
   ```
5. **Browser** → http://localhost:3000 → AI Incident Review tab → confirm green CONNECTED badge.
6. **Walk-in test (you, subject):** walk into frame empty-handed. Operator confirms `person` alert fires in browser. Click DISMISS to clear.

If all 6 steps pass → start the test matrix. If any fail → fix before proceeding (don't start test with broken plumbing).

---

## Test matrix

Run each scenario **3 times** with slight variation each time. Total ≈ 30 minutes for the full matrix.

| # | Scenario | Subject does | Watching for |
|---|----------|--------------|--------------|
| **1** | Person baseline | Walk through frame at normal pace | `person` fires every time, conf > 0.6 |
| **2** | Person stationary | Stand in frame for 10 sec, then move | `person` fires, conf stays high while still |
| **3** | Person at distance | Stand 5m / 10m / 20m from phone (3 trials) | At what distance does conf drop below 0.4? |
| **4** | Toy rifle visible | Walk through frame with rifle held visibly | `weapon_person` fires? Conf? |
| **5** | Toy rifle slung | Carry rifle slung over shoulder | Detection vs upright-carry |
| **6** | Toy rifle hidden | Hold rifle behind back / at side | Should NOT fire weapon_person; might fire person only |
| **7** | Backpack on shoulder | Walk normally with backpack | Probably just `person`; document if `explosive_device` fires |
| **8** | Bag dropped + walk away | Drop bag in frame, walk out | Static `explosive_device` detection on abandoned bag |
| **9** | Suspicious carry | Cradle bag in arms, look around | Test for `explosive_device` |
| **10** | Lighter held up | Hold lighter flame visibly (only if safe) | `fire` detection at distance |
| **11** | Multiple objects | Carry 2+ props at once | Class prioritisation |
| **12** | False-positive sweep | Walk past empty-handed wearing different clothes/jacket | False weapon_person alerts? |

**Operator job per row:** for each detected incident, classify in the browser:
- `SAFE` if it matches what the subject was carrying AND looks legitimate (e.g. authorised officer holding training weapon)
- `THREAT` if it matches what the subject was carrying AND is the threat scenario
- `DISMISS` if it's a false positive (no real object of that class)

---

## Recording template

Print or copy this into the notebook. One row per detected incident.

| Tick # | Scenario # | Time | Class fired | Confidence | Operator decision | Notes (lighting / angle / distance / odd behaviour) |
|--------|------------|------|-------------|------------|-------------------|------------------------------------------------------|
| 1 | 1 | 09:42:15 | person | 0.87 | DISMISS | clear sun, 8m, walking slow |
| 2 | 4 | 09:48:03 | weapon_person | 0.55 | THREAT | rifle held two-handed, shoulder height |
| ... | | | | | | |

Also note **misses**: scenarios where you expected an alert and didn't get one. Format: `MISS — scenario 4, trial 2, rifle visible, no alert`.

---

## Success criteria

| Class | Floor (acceptable) | Stretch (great) |
|-------|--------------------|-----------------|
| person | fires on >90% of walk-throughs | fires at every angle and distance < 20m |
| weapon_person | fires on >50% of toy-rifle-visible scenarios | fires consistently, conf > 0.5 |
| explosive_device | fires at least once across all bag scenarios | fires on abandoned-bag scenario at conf > 0.4 |
| fire | n/a (skip if no safe ignition) | fires on visible flame at < 10m |
| Audit chain | every classified incident produces a row in `decisions.jsonl` AND a clip in `incidents/` | replay clip in browser shows the moment of detection |

If we hit "Floor" for person + weapon_person, the test is a clear win for the CM pitch (we have validated drone-perspective detection of the headline classes in real-world conditions).

---

## Common failure modes & fixes

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Dashboard shows DISCONNECTED | AI workflow not running, or wrong port | Restart workflow, check `--gcs-port 8765` |
| No video in cv2 window | Phone not on same WiFi, or wrong URL | `ping <phone-ip>`, restart IP Webcam server |
| Video lags badly | Network too slow for HD stream | In IP Webcam: Video preferences → reduce resolution to 640x480 |
| `person` fires on background (parked car, lamppost) | False positive at low conf threshold | Bump `--alert-threshold` to 0.50 (won't catch low-conf weapon_person but reduces noise) |
| Phone overheats / dies | Outdoor sun + long capture | Shade the phone, plug into power bank if possible |
| Laptop disconnects from WiFi | Range issue if laptop is far from router | Move laptop closer, or have operator hold both devices on the balcony |

---

## Post-test (back at the workstation)

1. **Save the audit dir.** Rename `audit_runs/field_test_<ts>/` to something memorable (e.g. `audit_runs/balcony_test_2026_05_10/`).
2. **Quick review.** Open `decisions.jsonl` in a text editor — count incidents by class, by decision. Spot-check 2-3 incident clips in `incidents/` to confirm video quality.
3. **Write findings** as `reports/field_test_<date>.md`. Template:
   ```
   # Field test, 2026-05-10
   - Total incidents: N
   - By class: person=X, weapon_person=Y, explosive_device=Z
   - Mean confidence per class
   - Classification breakdown (SAFE / THREAT / DISMISS counts)
   - Notable misses (scenarios where alert didn't fire)
   - Notable false positives
   - Conditions: weather, time, lighting, altitude estimate
   - Verdict: ready for CM pitch / need more training data
   ```
4. **Pick 2-3 best clips** for the CM demo deck. These become the "look, the AI works in real-world conditions" payoff slide.
5. **Decide:** do the results justify the CM pitch as-is, or do we need a fast retraining pass before the meeting?

---

## What "good enough for CM" looks like

After this test, you should be able to make these claims credibly:

- "We deployed our police-trained AI on a phone-equivalent edge sensor and operated it from a real balcony at typical urban-overwatch altitude."
- "It detected people walking by with X% reliability."
- "When a person carried a visible weapon, it flagged the threat with Y% confidence in Z seconds, allowing the operator to classify it as authorised or unauthorised."
- "The full evidence chain — detection, alert, operator decision, video clip — was captured for every incident."
- "Here are 2-3 example clips."

If you can say all five honestly after the test, the AI side of the pitch is ready.
