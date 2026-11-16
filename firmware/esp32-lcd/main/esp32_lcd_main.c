#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include "driver/spi_master.h"
#include "driver/gpio.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_ops.h"
#include "esp_lcd_panel_vendor.h"
#include "esp_heap_caps.h"
#include "font8x16.h"

#define PIN_LCD_SCLK   12
#define PIN_LCD_MOSI   11
#define PIN_LCD_CS     10
#define PIN_LCD_DC      8
#define PIN_LCD_RST     9
#define PIN_LCD_BL     16
#define LCD_WIDTH   240
#define LCD_HEIGHT  240
#define UART_PORT   UART_NUM_0
#define UART_BAUD   115200
#define RX_BUF  64

static char rx_line[RX_BUF]; static int rx_len = 0;
static char cur_label[32] = "READY";
static float cur_score = 0.0f;
static bool cur_valid = false, dirty = true;
static esp_lcd_panel_handle_t lcd = NULL;

static inline uint16_t rgb565(uint8_t r, uint8_t g, uint8_t b) {
    return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
}
#define SWAP16(x) ((uint16_t)(((x)>>8)|((x)<<8)))

static void lcd_fill(int x1,int y1,int x2,int y2,uint16_t c) {
    if(x1>x2){int t=x1;x1=x2;x2=t;} if(y1>y2){int t=y1;y1=y2;y2=t;}
    uint32_t w=x2-x1+1,h=y2-y1+1,n=w*h;
    uint16_t *b=heap_caps_malloc(n*2,MALLOC_CAP_DMA); if(!b)return;
    uint16_t s=SWAP16(c);
    for(uint32_t i=0;i<n;i++)b[i]=s;
    esp_lcd_panel_draw_bitmap(lcd,x1,y1,x2+1,y2+1,b);
    free(b);
}

static void draw_char(int x,int y,uint8_t ch,int sc,uint16_t fg,uint16_t bg) {
    if(ch>=128)ch=' ';
    const uint8_t*gl=font8x16[ch];
    uint32_t sw=8*sc,sh=16*sc,n=sw*sh;
    uint16_t *b=heap_caps_malloc(n*2,MALLOC_CAP_DMA); if(!b)return;
    uint16_t fw=SWAP16(fg),bw=SWAP16(bg);
    for(int rw=0;rw<16;rw++){
        uint8_t bits=gl[rw];
        for(int r=0;r<sc;r++)
            for(int c=0;c<8;c++){
                uint16_t v=(bits&(1<<(7-c)))?fw:bw;
                for(int s=0;s<sc;s++)
                    b[(rw*sc+r)*sw+(c*sc+s)]=v;
            }
    }
    esp_lcd_panel_draw_bitmap(lcd,x,y,x+(int)sw,y+(int)sh,b);
    free(b);
}
static void draw_str(int x,int y,const char*s,int sc,uint16_t fg,uint16_t bg) {
    int cx=x,st=8*sc+(sc>1?sc:1);
    while(*s&&cx+8*sc<LCD_WIDTH){draw_char(cx,y,(uint8_t)*s,sc,fg,bg);cx+=st;s++;}
}

static void lcd_init(void) {
    gpio_set_direction(PIN_LCD_BL,GPIO_MODE_OUTPUT);
    gpio_set_level(PIN_LCD_BL,1);

    spi_bus_config_t bus={
        .mosi_io_num=PIN_LCD_MOSI,.miso_io_num=-1,.sclk_io_num=PIN_LCD_SCLK,
        .quadwp_io_num=-1,.quadhd_io_num=-1,.max_transfer_sz=LCD_WIDTH*80*2
    };
    ESP_ERROR_CHECK(spi_bus_initialize(SPI2_HOST,&bus,SPI_DMA_CH_AUTO));

    esp_lcd_panel_io_handle_t io=NULL;
    esp_lcd_panel_io_spi_config_t iocfg={
        .dc_gpio_num=PIN_LCD_DC,.cs_gpio_num=PIN_LCD_CS,
        .pclk_hz=26*1000*1000,.spi_mode=0,.trans_queue_depth=10,
        .lcd_cmd_bits=8,.lcd_param_bits=8
    };
    ESP_ERROR_CHECK(esp_lcd_new_panel_io_spi(SPI2_HOST,&iocfg,&io));

    esp_lcd_panel_dev_config_t pcfg={
        .reset_gpio_num=PIN_LCD_RST,
        .rgb_ele_order=LCD_RGB_ELEMENT_ORDER_RGB,
        .bits_per_pixel=16,
    };
    ESP_ERROR_CHECK(esp_lcd_new_panel_st7789(io,&pcfg,&lcd));
    ESP_ERROR_CHECK(esp_lcd_panel_reset(lcd));
    ESP_ERROR_CHECK(esp_lcd_panel_init(lcd));
    ESP_ERROR_CHECK(esp_lcd_panel_disp_on_off(lcd,true));
}

static void parse_protocol(const char *d,int len) {
    (void)len;
    // 纯文本协议：整行即标签内容，过滤不可打印/控制字符
    char c[64];int ci=0;
    while(*d&&ci<62){
        char ch=*d;
        if(ch>=32&&ch<127){c[ci++]=ch;}  // 仅保留可打印 ASCII
        d++;
    }
    c[ci]=0;
    if(ci==0||strcmp(c,"none")==0){cur_valid=false;strcpy(cur_label,"---");cur_score=0;dirty=true;return;}
    strncpy(cur_label,c,31);cur_label[31]=0;
    cur_score=0;cur_valid=true;dirty=true;
}

static void update_display(void) {
    if(!dirty)return;dirty=false;
    lcd_fill(0,0,LCD_WIDTH-1,LCD_HEIGHT-1,rgb565(0,0,255));
    lcd_fill(10,10,50,50,rgb565(255,0,0));
    lcd_fill(180,180,230,230,rgb565(0,255,0));
    draw_str(20,100,cur_label,2,rgb565(255,255,255),0x0000);
    char s[16];snprintf(s,sizeof(s),"%.2f",cur_score);
    draw_str(20,140,s,1,rgb565(255,255,0),0x0000);
}

void app_main(void) {
    uart_config_t u={.baud_rate=UART_BAUD,.data_bits=UART_DATA_8_BITS,.parity=UART_PARITY_DISABLE,
        .stop_bits=UART_STOP_BITS_1,.flow_ctrl=UART_HW_FLOWCTRL_DISABLE,.source_clk=UART_SCLK_DEFAULT};
    uart_param_config(UART_PORT,&u);
    uart_set_pin(UART_PORT,UART_PIN_NO_CHANGE,UART_PIN_NO_CHANGE,UART_PIN_NO_CHANGE,UART_PIN_NO_CHANGE);
    uart_driver_install(UART_PORT,256,0,0,NULL,0);
    lcd_init();
    cur_valid=true;strcpy(cur_label,"READY");dirty=true;
    TickType_t last=xTaskGetTickCount();
    while(1){
        uint8_t tmp[64];int n=uart_read_bytes(UART_PORT,tmp,63,0);
        if(n>0){for(int i=0;i<n;i++){char c=(char)tmp[i];
            if(c=='\n'){rx_line[rx_len]=0;parse_protocol(rx_line,rx_len);rx_len=0;}
            else if(c!='\r'){if(rx_len<RX_BUF-1)rx_line[rx_len++]=c;else rx_len=0;}}}
        TickType_t now=xTaskGetTickCount();
        if((now-last)*portTICK_PERIOD_MS>=200){update_display();last=now;}
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}
