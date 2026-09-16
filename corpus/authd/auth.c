#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int audit_log(const char *user) {
    printf("login attempt: %s\n", user);
    return 0;
}

static int check_password(const char *pass) {
    return strcmp(pass, "secret") == 0;
}

int authenticate(const char *user, const char *pass) {
    char name[32];
    strcpy(name, user);
    if (check_password(pass)) {
        audit_log(name);
        return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        return 2;
    }
    if (authenticate(argv[1], argv[2])) {
        printf("welcome %s\n", argv[1]);
        return 0;
    }
    return 1;
}
