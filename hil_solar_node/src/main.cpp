/*
 * Solar-powered adaptive sensing node: hardware-in-the-loop version (record +
 * replay)
 * ----------------------------------------------------------------
 * Board : NodeMCU ESP8266  (PlatformIO: src/main.cpp)
 * Wiring: DHT11 DATA -> D5, soil sensor AO -> A0, 3V3 / GND as in the diagram.
 *
 * What is REAL : the DHT11 (temperature, humidity) and the soil-moisture
 * readings. What is SIMULATED (same model as the ngspice co-simulation):
 *      - the weather / solar panel current
 *      - the battery (10 mAh, 3.0-4.2 V, modelled as a 30 F capacitor)
 *      - the energy cost of every sample and transmission
 * Four algorithms run on the real readings and drain/charge the virtual
 * battery: 0 fixed_rate   1 signal_only   2 energy_only   3 combined_v4
 *
 * FAIR BENCHMARK = RECORD ONCE, REPLAY FOR EVERY RUN
 *      The real sensor is read live ONLY while recording (command c). The
 * recorded trace is then replayed (at full speed) for every weather x method x
 * seed run, so all methods see exactly the same signal and the same weather.
 *
 * Libraries (Arduino IDE -> Library Manager):
 *      "DHT sensor library" by Adafruit  +  "Adafruit Unified Sensor"
 * Board package: ESP8266 by ESP8266 Community.  Board: "NodeMCU 1.0 (ESP-12E
 * Module)". Serial Monitor: 115200 baud, line ending "Newline".
 *
 * Serial commands (send one, press Enter):
 *      w0..w3  weather: 0 sunny, 1 sunny->cloudy, 2 cloudy->sunny, 3 cloudy
 *      m0..m3  method : 0 fixed_rate, 1 signal_only, 2 energy_only, 3
 * combined_v4 g0..g2  signal : 0 temperature, 1 humidity, 2 soil moisture
 * (discards any recorded trace) s<n>    weather random seed (default 1),
 * restarts the run c       RECORD the real sensor for one full run (prints
 * TRACE lines), then switch to replay r       restart the current run a sweep
 * ALL 16 (weather x method) combinations with the current seed (needs a
 * recorded trace) b       sweep all 16 combinations for seeds 1..N_SEEDS (needs
 * a recorded trace) h       help Output lines start with  RUN, LOG, RESULT,
 * TRACE  so a script can pick them out.
 *
 * Energy values are model numbers (joules of a 10 mAh cell), not measured on
 * the ESP.
 */
#include <Arduino.h>
#include <DHT.h>
#include <math.h>

// ============================ user settings ============================
#define DHT_PIN D5
#define DHT_TYPE DHT11
#define SOIL_PIN A0

const uint32_t TICK_MS =
    70; // real ms per tick WHILE RECORDING. 5760 ticks x 70 ms = ~6.7 min per
        // signal (3 signals ~ 20 min). DHT11 itself refreshes every 2 s, so
        // each DHT reading repeats for ~28 ticks (soil is read every tick).
        // 1000 -> ~96 min per signal (finer trace). Replay ignores this and
        // runs at full speed.
const float DT = 30.0f; // simulated seconds per tick (one "row")
const int DAYS = 2;
const int N = (int)(DAYS * 86400 / DT); // 5760 ticks
const bool LOG_CSV =
    false; // false = readable log lines, true = compact CSV (for plotting)
const int LOG_EVERY =
    20;            // print a LOG line every 20 ticks (= 10 simulated minutes)
uint32_t SEED = 1; // same weather for every method => fair comparison (change
                   // with command s<n>)
const int N_SEEDS = 5; // command b sweeps seeds 1..N_SEEDS

const float SOIL_DRY_RAW =
    1024.0f; // analogRead with the probe in air   (calibrate!)
const float SOIL_WET_RAW =
    400.0f; // analogRead with the probe in water (calibrate!)

// battery / panel / load model  (identical to sim_core.py Cfg)
const float BATT_MAH = 10.0f, V_EMPTY = 3.0f, V_FULL = 4.2f, V_ALIVE = 3.05f,
            SOC0 = 0.5f;
const float ISC = 6e-3f;       // panel short-circuit current at full sun (A)
const float P_SLEEP = 1.5e-4f; // W  sleeping node  (MEASURE yours and edit!)
const float E_SAMPLE = 0.14f;  // J  per sample (0.14 W x 1 s)
const float E_TX = 0.55f;      // J  per transmission (0.55 W x 1 s)
const float CAP_F = BATT_MAH * 3.6f / (V_FULL - V_EMPTY);
const float SUNRISE = 6.5f, SUNSET = 17.5f;

// policy parameters
const int FIXED_INTERVAL = 10;
const int MAX_STALE = 120; // combined_v4 mandatory row spacing
const float SOC_FLOOR = 0.15f, SOC_REF = 0.6f, TX_FRAC = 0.02f;
const int MAX_INTERVAL = 30;
const float U1 = 0.025f * 0.2f, U2 = 0.08f * 0.2f, U3 = 0.2f * 0.2f,
            U4 = 0.5f * 0.2f;

// ============================ globals ============================
DHT dht(DHT_PIN, DHT_TYPE);

static int16_t vals[N];     // ground-truth readings x100
static uint8_t flags[N];    // bit0 acquired, bit1 transmitted
static int16_t refTrace[N]; // recorded real trace x100 (replayed for every run)

const char *W_NAME[4] = {"sunny", "sunny_to_cloudy", "cloudy_to_sunny",
                         "cloudy"};
const char *M_NAME[4] = {"fixed_rate", "signal_only", "energy_only",
                         "combined_v4"};
const char *G_NAME[3] = {"temperature", "humidity", "soil"};
const bool W_SUNNY[4][2] = {
    {true, true}, {true, false}, {false, true}, {false, false}};

int weatherSel = 0, methodSel = 3, signalSel = 0;
bool running = false, sweep = false, sweepSeeds = false;
bool capture = false, replay = false, haveRef = false;
int sweepIdx = 0;
int tickIdx = 0;
uint32_t lastTickMs = 0;

// battery + stats
float V, cloudX, eIn, eClip, eLoad;
int aliveTicks, brownouts, nAcq, nTx;
bool wasAlive;
float minSoc;

// last good sensor values
float lastT = 22.0f, lastH = 50.0f, lastS = 50.0f;

// policy state
int nextIndex;
float obs[10];
int nobs;
bool hasTx;
float lastTx;
bool haveLH;
float lowV, highV;
const int RING = 80;
float rt[RING], rv[RING], rs[RING];
int rpos, rn;

// ============================ helpers ============================
static float clampf(float x, float a, float b) {
  return x < a ? a : (x > b ? b : x);
}

static float randn() {
  float u1 = (float)random(1, 100000) / 100000.0f;
  float u2 = (float)random(1, 100000) / 100000.0f;
  return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

static float clearSky(float hour) {
  float x = (hour - SUNRISE) / (SUNSET - SUNRISE);
  if (x <= 0.0f || x >= 1.0f)
    return 0.0f;
  return powf(sinf(PI * x), 1.2f);
}

// irradiance 0..1 for tick i (advances the cloud random process once per tick)
static float irradiance(int i) {
  float t = i * DT;
  int day = (int)(t / 86400.0f);
  if (day > 1)
    day = 1;
  cloudX = 0.97f * cloudX + 0.22f * randn();
  float cloud;
  if (W_SUNNY[weatherSel][day])
    cloud = clampf(0.97f + 0.02f * cloudX, 0.88f, 1.0f);
  else
    cloud = clampf(0.15f * expf(0.55f * cloudX), 0.03f, 0.40f);
  float hour = fmodf(t / 3600.0f, 24.0f);
  return clearSky(hour) * cloud;
}

static float medianOf(const float *a, int n) {
  float b[16];
  for (int i = 0; i < n; i++)
    b[i] = a[i];
  for (int i = 1; i < n; i++) {
    float k = b[i];
    int j = i - 1;
    while (j >= 0 && b[j] > k) {
      b[j + 1] = b[j];
      j--;
    }
    b[j + 1] = k;
  }
  return (n & 1) ? b[n / 2] : 0.5f * (b[n / 2 - 1] + b[n / 2]);
}

static float readSignal() {
  if (signalSel == 0) {
    float t = dht.readTemperature();
    if (!isnan(t))
      lastT = t;
    return lastT;
  }
  if (signalSel == 1) {
    float h = dht.readHumidity();
    if (!isnan(h))
      lastH = h;
    return lastH;
  }
  float raw = (float)analogRead(SOIL_PIN);
  lastS = clampf((SOIL_DRY_RAW - raw) / (SOIL_DRY_RAW - SOIL_WET_RAW) * 100.0f,
                 0.0f, 100.0f);
  return lastS;
}

// ============================ policies ============================
static bool isMand(int i) {
  return (i % MAX_STALE == 0 && i < N - 1) || i == N - 1;
}
static int nxtMand(int i) {
  int k = (i / MAX_STALE + 1) * MAX_STALE;
  if (k < N - 1)
    return k;
  return (i < N - 1) ? N - 1 : N;
}
static int ringIdx(int k) {
  return (rpos - 1 - k + 2 * RING) % RING;
} // k = 0 newest

// returns true if the node acquires a sample on this tick; 'send' = also
// transmit
static bool policyStep(int i, float v, float t, float soc, bool &send) {
  send = false;
  if (methodSel == 0) { // fixed_rate
    nextIndex = i + FIXED_INTERVAL;
    send = true;
    return true;
  }
  if (methodSel == 1 || methodSel == 2) { // signal_only / energy_only
    int interval;
    if (methodSel == 1) {
      if (nobs < 10)
        obs[nobs++] = v;
      else {
        for (int k = 0; k < 9; k++)
          obs[k] = obs[k + 1];
        obs[9] = v;
      }
      float raw = 0.0f;
      if (nobs > 1) {
        for (int k = 1; k < nobs; k++)
          raw += fabsf(obs[k] - obs[k - 1]);
        raw /= (nobs - 1);
      }
      interval = raw >= 0.8f ? 1 : (raw <= 0.15f ? 30 : 10);
    } else {
      float b = soc * 100.0f;
      interval = b < 30.0f ? 30 : (b < 70.0f ? 10 : 5);
    }
    send = (!hasTx) || fabsf(v - lastTx) >= 0.3f;
    if (send) {
      lastTx = v;
      hasTx = true;
    }
    nextIndex = i + interval;
    return true;
  }
  // ---- combined_v4 (harvest-aware port of main.py) ----
  bool mand = isMand(i);
  if (!mand && soc < SOC_FLOOR) {
    nextIndex = nxtMand(i);
    return false;
  }
  if (!haveLH) {
    lowV = highV = v;
    haveLH = true;
  } else {
    lowV = fminf(lowV, v);
    highV = fmaxf(highV, v);
  }
  float scale =
      fmaxf(fmaxf(highV - lowV, 1e-3f * fmaxf(fabsf(v), 1.0f)), 1e-9f);
  float var = 0.0f, pred = 0.0f;
  if (rn >= 2) {
    int m = rn < 10 ? rn : 10;
    float rates[10], dts[10];
    int nd = 0;
    for (int k = m - 1; k >= 1; k--) {
      int a = ringIdx(k), b = ringIdx(k - 1);
      float dtt = fmaxf(rt[b] - rt[a], 1e-9f);
      rates[nd] = fabsf(rv[b] - rv[a]) / dtt;
      dts[nd] = dtt;
      nd++;
    }
    var = medianOf(rates, nd) * medianOf(dts, nd) / scale;
    int n0 = ringIdx(0), n1 = ringIdx(1);
    float slope = (rv[n0] - rv[n1]) / fmaxf(rt[n0] - rt[n1], 1e-9f);
    float st = clampf(slope * (t - rt[n0]), -0.5f * scale, 0.5f * scale);
    pred = fabsf(v - (rv[n0] + st)) / scale;
  }
  float urg = fmaxf(var, pred);
  int desired = urg >= U4   ? 1
                : urg >= U3 ? 3
                : urg >= U2 ? 6
                : urg >= U1 ? 12
                            : MAX_INTERVAL;

  float r = soc / SOC_REF;
  for (int k = 0; k < rn;
       k++) { // battery rising over the last >=30 min => surplus
    int ix = ringIdx(k);
    if (t - rt[ix] >= 1800.0f) {
      if (soc - rs[ix] > 0.01f)
        r += 0.3f;
      break;
    }
  }
  int flr = r >= 1.0f   ? 1
            : r >= 0.8f ? 3
            : r >= 0.6f ? 6
            : r >= 0.4f ? 12
                        : MAX_INTERVAL;
  int interval;
  if (urg >= U4 && r >= 0.6f) {
    int mn = flr < 3 ? flr : 3;
    interval = desired > mn ? desired : mn;
  } else {
    interval = desired > flr ? desired : flr;
  }
  if (interval < 1)
    interval = 1;
  if (interval > MAX_INTERVAL)
    interval = MAX_INTERVAL;

  if (mand)
    send = true;
  else {
    float thr = TX_FRAC * scale *
                clampf(1.0f / fmaxf(fminf(r, 1.0f), 1e-3f), 1.0f, 4.0f);
    if (urg >= U2)
      thr *= 0.5f;
    send = ((!hasTx) || fabsf(v - lastTx) >= thr) && soc >= SOC_FLOOR;
  }
  if (send) {
    lastTx = v;
    hasTx = true;
  }
  rt[rpos] = t;
  rv[rpos] = v;
  rs[rpos] = soc;
  rpos = (rpos + 1) % RING;
  if (rn < RING)
    rn++;
  int nm = nxtMand(i), cand = i + interval;
  nextIndex = cand < nm ? cand : nm;
  return true;
}

// ============================ run control ============================
static void startRun() {
  tickIdx = 0;
  nextIndex = 0;
  nobs = 0;
  hasTx = false;
  lastTx = 0;
  haveLH = false;
  rpos = 0;
  rn = 0;
  V = V_EMPTY + (V_FULL - V_EMPTY) * SOC0;
  cloudX = 0;
  eIn = eClip = eLoad = 0;
  aliveTicks = 0;
  brownouts = 0;
  nAcq = 0;
  nTx = 0;
  wasAlive = true;
  minSoc = 1.0f;
  for (int i = 0; i < N; i++) {
    vals[i] = 0;
    flags[i] = 0;
  }
  randomSeed(SEED + 1); // identical weather for every method
  running = true;
  lastTickMs = millis();
  Serial.printf("RUN,%s,%s,%s\n", W_NAME[weatherSel], G_NAME[signalSel],
                M_NAME[methodSel]);
}

static void printResult() {
  // gateway reconstruction: linear interpolation between transmitted samples
  int first = -1, last = -1;
  for (int i = 0; i < N; i++)
    if (flags[i] & 2) {
      if (first < 0)
        first = i;
      last = i;
    }
  float nrmse = NAN, nmae = NAN, maxGap = (float)N * DT / 60.0f;
  if (first >= 0) {
    float vmin = vals[0], vmax = vals[0];
    for (int i = 0; i < N; i++) {
      float x = vals[i];
      if (x < vmin)
        vmin = x;
      if (x > vmax)
        vmax = x;
    }
    float span = fmaxf((vmax - vmin) / 100.0f, 1e-12f);
    double se = 0, ae = 0;
    int prev = first;
    int maxg = first;
    for (int i = 0; i < first; i++) {
      double e = (vals[first] - vals[i]) / 100.0;
      se += e * e;
      ae += fabs(e);
    }
    for (int i = first + 1; i <= last; i++) {
      if (flags[i] & 2) {
        for (int k = prev; k < i; k++) {
          double p = vals[prev] +
                     (double)(vals[i] - vals[prev]) * (k - prev) / (i - prev);
          double e = (p - vals[k]) / 100.0;
          se += e * e;
          ae += fabs(e);
        }
        if (i - prev > maxg)
          maxg = i - prev;
        prev = i;
      }
    }
    {
      double e = (vals[last] - vals[last]) / 100.0;
      se += e * e;
      ae += fabs(e);
    }
    for (int i = last + 1; i < N; i++) {
      double e = (vals[last] - vals[i]) / 100.0;
      se += e * e;
      ae += fabs(e);
    }
    nrmse = (float)(sqrt(se / N) / span * 100.0);
    nmae = (float)(ae / N / span * 100.0);
    int lead = first, trail = N - 1 - last;
    if (lead > maxg)
      maxg = lead;
    if (trail > maxg)
      maxg = trail;
    maxGap = maxg * DT / 60.0f;
  }
  float socEnd = clampf((V - V_EMPTY) / (V_FULL - V_EMPTY), 0.0f, 1.0f);
  Serial.printf("\n==== RUN FINISHED: weather=%s | signal=%s | method=%s | "
                "seed=%lu ====\n",
                W_NAME[weatherSel], G_NAME[signalSel], M_NAME[methodSel],
                (unsigned long)SEED);
  Serial.printf(
      "  Gateway reconstruction error (NRMSE): %.2f %%   (lower is better)\n",
      nrmse);
  Serial.printf("  Samples taken: %d of %d    Transmissions: %d\n", nAcq, N,
                nTx);
  Serial.printf("  Energy used by node: %.1f J   harvested from sun: %.1f J   "
                "wasted (battery full): %.1f J\n",
                eLoad, eIn, eClip);
  Serial.printf("  Node uptime: %.1f %%    Brown-outs: %d\n",
                100.0f * aliveTicks / N, brownouts);
  Serial.printf("  Battery: lowest %.1f %%    final %.1f %%\n", 100.0f * minSoc,
                100.0f * socEnd);
  Serial.printf("  Longest time without data reaching the gateway: %.1f min\n",
                maxGap);
  Serial.println("  (next line = same numbers as CSV for collect_serial.py)");
  Serial.printf("RESULT,%s,%s,%s,%.3f,%.3f,%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%.2f,%."
                "2f,%.1f,%lu\n",
                W_NAME[weatherSel], G_NAME[signalSel], M_NAME[methodSel], nrmse,
                nmae, nAcq, nTx, eLoad, eIn, eClip, 100.0f * aliveTicks / N,
                brownouts, 100.0f * minSoc, 100.0f * socEnd, maxGap,
                (unsigned long)SEED);
}

static void nextSweep() {
  int total = sweepSeeds ? 16 * N_SEEDS : 16;
  if (sweepIdx >= total) {
    sweep = false;
    sweepSeeds = false;
    Serial.println("SWEEP_DONE");
    return;
  }
  int k = sweepIdx % 16;
  if (sweepSeeds)
    SEED = 1 + sweepIdx / 16;
  weatherSel = k / 4;
  methodSel = k % 4;
  sweepIdx++;
  startRun();
}

static const char *unitOf() {
  return signalSel == 0 ? "C" : (signalSel == 1 ? "%RH" : "%");
}

static void logLine(int i, float t, float G, float soc, float v, bool alive) {
  if (LOG_CSV) {
    Serial.printf("LOG,%.2f,%.3f,%.3f,%.1f,%.2f,%d,%d,%d\n", t / 3600.0f, G, V,
                  100.0f * soc, v, nAcq, nTx, alive ? 1 : 0);
    return;
  }
  int day = (int)(t / 86400.0f) + 1;
  int sod = (int)fmodf(t, 86400.0f);
  Serial.printf("day %d %02d:%02d | sun %3d%% | battery %.3f V (%5.1f%%) | %s "
                "%.1f %s | samples %4d | sent %4d | node %s\n",
                day, sod / 3600, (sod % 3600) / 60, (int)(G * 100.0f + 0.5f), V,
                100.0f * soc, G_NAME[signalSel], v, unitOf(), nAcq, nTx,
                alive ? "ON" : "OFF (battery low)");
}

static void tick() {
  int i = tickIdx;
  float t = i * DT;
  float G = irradiance(i);
  // live sensor while recording; recorded trace (identical for every run) while
  // replaying
  float v = (replay && haveRef) ? refTrace[i] / 100.0f : readSignal();
  vals[i] = (int16_t)lroundf(v * 100.0f);
  if (capture)
    Serial.printf("TRACE,%s,%d,%.2f\n", G_NAME[signalSel], i, v);

  float soc = clampf((V - V_EMPTY) / (V_FULL - V_EMPTY), 0.0f, 1.0f);
  bool alive = V > V_ALIVE;
  bool acq = false, send = false;
  if (alive && i >= nextIndex)
    acq = policyStep(i, v, t, soc, send);

  float iChg = ISC * G;
  float eSleep = alive ? P_SLEEP * DT : 0.0f;
  float eEvent = acq ? (E_SAMPLE + (send ? E_TX : 0.0f)) : 0.0f;
  eIn += V * iChg * DT;
  eLoad += eSleep + eEvent;
  V += (iChg * DT - (eSleep + eEvent) / fmaxf(V, 2.5f)) / CAP_F;
  if (V > V_FULL) {
    eClip += (V - V_FULL) * CAP_F * V_FULL;
    V = V_FULL;
  }
  if (V < 2.0f)
    V = 2.0f;

  if (acq) {
    flags[i] |= 1;
    nAcq++;
    if (send) {
      flags[i] |= 2;
      nTx++;
    }
  }
  if (alive)
    aliveTicks++;
  bool aliveNow = V > V_ALIVE;
  if (wasAlive && !aliveNow)
    brownouts++;
  wasAlive = aliveNow;
  float socNow = clampf((V - V_EMPTY) / (V_FULL - V_EMPTY), 0.0f, 1.0f);
  if (socNow < minSoc)
    minSoc = socNow;

  if (!replay && i % LOG_EVERY == 0)
    logLine(i, t, G, socNow, v,
            alive); // no LOG lines while replaying: keeps sweeps fast

  tickIdx++;
  if (tickIdx >= N) {
    running = false;
    if (capture) { // recording finished: keep the trace, switch to replay
      memcpy(refTrace, vals, sizeof(vals));
      capture = false;
      haveRef = true;
      replay = true;
      Serial.printf("CAPTURE_DONE,%s\n", G_NAME[signalSel]);
    } else
      printResult();
    if (sweep)
      nextSweep();
  }
}

static void printHelp() {
  Serial.println(F("LOG LINE: day + simulated clock | sun = sunlight 0-100% | "
                   "battery = virtual battery volts and charge % | sensor = "
                   "real reading | samples/sent = running totals | node = ON "
                   "while battery above brown-out"));
  Serial.println(F("HELP,Commands: w0-3 weather | m0-3 method | g0-2 signal | "
                   "s<n> seed | c record real trace | r restart | a sweep 16 | "
                   "b sweep 16 x seeds | h help"));
  Serial.println(
      F("HEADER,weather,signal,method,nrmse%,nmae%,samples,tx,load_J,harvest_J,"
        "clipped_J,uptime%,brownouts,minSOC%,finalSOC%,max_gap_min,seed"));
}

static void handleCommand(String s) {
  s.trim();
  if (s.length() == 0)
    return;
  char c = s[0];
  int n = s.length() > 1 ? s.substring(1).toInt() : 0;
  if (c == 'w' && n >= 0 && n < 4) {
    weatherSel = n;
    sweep = false;
    startRun();
  } else if (c == 'm' && n >= 0 && n < 4) {
    methodSel = n;
    sweep = false;
    startRun();
  } else if (c == 'g' && n >= 0 && n < 3) {
    signalSel = n;
    haveRef = false;
    replay = false;
    capture = false;
    sweep = false;
    startRun();
  } else if (c == 's' && n > 0) {
    SEED = (uint32_t)n;
    sweep = false;
    startRun();
  } else if (c == 'c') {
    sweep = false;
    capture = true;
    replay = false;
    startRun();
  } else if (c == 'r') {
    sweep = false;
    startRun();
  } else if (c == 'a' || c == 'b') {
    if (!haveRef)
      Serial.println("ERR,record a real trace first (send c)");
    else {
      sweep = true;
      sweepSeeds = (c == 'b');
      sweepIdx = 0;
      nextSweep();
    }
  } else
    printHelp();
}

void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println(F("\nSolar adaptive-sensing node (hardware-in-the-loop)"));
  dht.begin();
  delay(2000);
  for (int k = 0; k < 3; k++) {
    dht.readTemperature();
    dht.readHumidity();
    delay(2100);
  }
  Serial.printf("free heap: %u bytes\n", (unsigned)ESP.getFreeHeap());
  printHelp();
  startRun();
}

void loop() {
  static String line;
  while (Serial.available()) {
    char ch = (char)Serial.read();
    if (ch == '\n' || ch == '\r') {
      handleCommand(line);
      line = "";
    } else
      line += ch;
  }
  uint32_t tm = replay ? 0 : TICK_MS; // replay runs at full speed
  if (running && (uint32_t)(millis() - lastTickMs) >= tm) {
    lastTickMs += tm;
    tick();
  }
}