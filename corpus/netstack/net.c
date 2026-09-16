#include <stdio.h>
#include <string.h>
#include <stdint.h>

struct packet {
    uint16_t len;
    char data[256];
};

static void handle_packet(const struct packet *pkt) {
    char buf[64];
    memcpy(buf, pkt->data, pkt->len);
    buf[63] = '\0';
    printf("payload: %s\n", buf);
}

int main(int argc, char **argv) {
    struct packet pkt;
    if (argc < 2) {
        return 2;
    }
    pkt.len = (uint16_t)strlen(argv[1]);
    memset(pkt.data, 0, sizeof(pkt.data));
    strncpy(pkt.data, argv[1], sizeof(pkt.data) - 1);
    handle_packet(&pkt);
    return 0;
}
