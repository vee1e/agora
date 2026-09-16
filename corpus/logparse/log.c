#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int emit(const char *level, const char *msg) {
    char line[48];
    snprintf(line, sizeof(line), "[%s] %s", level, msg);
    puts(line);
    return 0;
}

void parse_line(const char *raw) {
    char field[32];
    const char *sep = strchr(raw, ':');
    if (sep == NULL) {
        emit("warn", "malformed");
        return;
    }
    size_t n = (size_t)(sep - raw);
    memcpy(field, raw, n);
    field[n] = '\0';
    emit("info", field);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        return 2;
    }
    parse_line(argv[1]);
    return 0;
}
