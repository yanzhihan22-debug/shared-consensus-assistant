// ESP32-S3 + ST7789 LCD + 双舵机云台 — 融合固件
// 一条串口/UDP 同时接收：LCD文字 和 s1:xx s2:xx 舵机指令
#define LGFX_USE_V1
#include <LovyanGFX.hpp>
#include <WiFi.h>
#include <WiFiUdp.h>

// 请改成你自己的 WiFi；不要把真实密码提交到仓库
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
const int UDP_PORT = 8888;
WiFiUDP udp;

// ===== LCD =====
class LGFX_ESP32S3_LCD : public lgfx::LGFX_Device
{
  lgfx::Panel_ST7789  _panel_instance;
  lgfx::Bus_SPI       _bus_instance;
public:
  LGFX_ESP32S3_LCD(void) {
    { auto cfg = _bus_instance.config();
      cfg.spi_host = SPI2_HOST; cfg.spi_mode = 0;
      cfg.freq_write = 40000000; cfg.freq_read = 16000000;
      cfg.pin_sclk = 21; cfg.pin_mosi = 47; cfg.pin_miso = -1; cfg.pin_dc = 40;
      _bus_instance.config(cfg); _panel_instance.setBus(&_bus_instance); }
    { auto cfg = _panel_instance.config();
      cfg.pin_cs = 41; cfg.pin_rst = 45; cfg.pin_busy = -1;
      cfg.panel_width = 240; cfg.panel_height = 240;
      cfg.offset_x = 0; cfg.offset_y = 0;
      cfg.invert = true; cfg.rgb_order = false;
      _panel_instance.config(cfg); }
    setPanel(&_panel_instance);
  }
};
LGFX_ESP32S3_LCD lcd;

char last_display[300] = "";
char render_buf[300] = "";
unsigned long last_render = 0;
const unsigned long RENDER_MS = 200;

void draw_wrapped(const char *text, int &y, int size, int color, int max_chars) {
  lcd.setTextColor(color, TFT_BLACK);
  lcd.setTextSize(size);
  char buf[128];
  strcpy(buf, text);
  char *p = buf;
  while (*p) {
    int len = strlen(p);
    if (len > max_chars) {
      int break_at = max_chars;
      for (int i = max_chars; i >= 0; i--) {
        if (p[i] == ' ') { break_at = i; break; }
      }
      char saved = p[break_at];
      p[break_at] = 0;
      lcd.setCursor(10, y);
      lcd.print(p);
      p[break_at] = saved;
      p += break_at + 1;
      y += (size == 1) ? 14 : 24;
    } else {
      lcd.setCursor(10, y);
      lcd.print(p);
      p += len;
      y += (size == 1) ? 14 : 24;
    }
  }
}

void show(const char *text) {
  lcd.fillScreen(TFT_BLACK);
  int y = 20;
  char buf[300];
  strcpy(buf, text);
  char *line = strtok(buf, "|");
  while (line) {
    int sz = 2, clr = TFT_WHITE, maxc = 19;
    if (strstr(line, "stable") || strstr(line, "Shake")) {
      sz = 2; clr = TFT_YELLOW; maxc = 19;
    } else if (strstr(line, "LOCKED")) { clr = TFT_GREEN; }
    else if (strstr(line, "SHAKE"))  { clr = TFT_CYAN; }
    draw_wrapped(line, y, sz, clr, maxc);
    line = strtok(NULL, "|");
  }
}

// ===== 舵机 =====
#define S1_PIN  14
#define S2_PIN  13
#define SERVO_FREQ 50
#define SERVO_RES  12

int angleToDuty(int a) {
  a = constrain(a, 0, 180);
  return map(map(a, 0, 180, 500, 2400), 0, 20000, 0, 4095);
}

#define ALPHA 0.3
float s1_smooth = 60, s2_smooth = 120;

void parse_line(const char *line) {
  float v1, v2;
  if (sscanf(line, "s1:%f s2:%f", &v1, &v2) >= 2) {
    s1_smooth = s1_smooth * (1 - ALPHA) + v1 * ALPHA;
    s2_smooth = s2_smooth * (1 - ALPHA) + v2 * ALPHA;
    ledcWrite(S1_PIN, angleToDuty((int)s1_smooth));
    ledcWrite(S2_PIN, angleToDuty((int)s2_smooth));
  } else {
    strcpy(render_buf, line);
    last_render = 0;
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(42, OUTPUT); digitalWrite(42, HIGH);
  lcd.init(); lcd.setRotation(0); lcd.fillScreen(TFT_BLACK);
  ledcAttach(S1_PIN, SERVO_FREQ, SERVO_RES);
  ledcAttach(S2_PIN, SERVO_FREQ, SERVO_RES);
  ledcWrite(S1_PIN, angleToDuty(60));
  ledcWrite(S2_PIN, angleToDuty(120));

  lcd.setTextSize(1);
  lcd.setTextColor(TFT_WHITE, TFT_BLACK);
  lcd.setCursor(10, 40);
  lcd.print("Connecting WiFi...");

  WiFi.mode(WIFI_STA);
  WiFi.config(IPAddress(192,168,1,200), IPAddress(192,168,1,1), IPAddress(255,255,255,0));
  WiFi.begin(ssid, password);
  int tries = 0;
  while (WiFi.status() != WL_CONNECTED && tries < 40) {
    delay(500);
    tries++;
  }

  lcd.fillScreen(TFT_BLACK);
  lcd.setCursor(10, 40);
  if (WiFi.status() == WL_CONNECTED) {
    lcd.setTextColor(TFT_GREEN, TFT_BLACK);
    lcd.print("WiFi OK");
    lcd.setCursor(10, 70);
    lcd.setTextColor(TFT_WHITE, TFT_BLACK);
    lcd.print("IP: ");
    lcd.print(WiFi.localIP());
    udp.begin(UDP_PORT);
  } else {
    lcd.setTextColor(TFT_RED, TFT_BLACK);
    lcd.print("WiFi Failed");
    lcd.setCursor(10, 70);
    lcd.setTextColor(TFT_WHITE, TFT_BLACK);
    lcd.print("Using Serial only");
  }

  delay(2000);
  lcd.fillScreen(TFT_BLACK);
}

void loop() {
  // --- UDP 接收（LCD文字 + 舵机指令） ---
  int packetSize = udp.parsePacket();
  if (packetSize) {
    char buf[300];
    int len = udp.read(buf, min(packetSize, 299));
    if (len > 0) {
      buf[len] = 0;
      int slen = strlen(buf);
      while (slen > 0 && (buf[slen-1] == '\n' || buf[slen-1] == '\r'))
        buf[--slen] = 0;
      if (slen > 0) parse_line(buf);
    }
  }

  // --- 串口接收 ---
  while (Serial.available()) {
    char c = Serial.read();
    static int idx = 0;
    static char line[300];
    if (c == '\n') {
      line[idx] = 0;
      if (idx > 0) parse_line(line);
      idx = 0;
    } else if (c != '\r') {
      if (idx < 299) line[idx++] = c;
    }
  }

  // --- LCD 刷新 ---
  if (millis() - last_render >= RENDER_MS) {
    last_render = millis();
    if (strcmp(render_buf, last_display) != 0) {
      strcpy(last_display, render_buf);
      show(render_buf);
    }
  }

  delay(1);
}
