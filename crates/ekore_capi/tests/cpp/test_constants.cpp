#include <ekore_capi.h>
#include <iostream>
#include <cstdlib>

using namespace std;

static int check_val(const char *name, size_t actual, size_t expected)
{
    if (actual != expected) {
        cerr << "FAIL: " << name << " - expected " << expected << ", got " << actual << endl;
        return 1;
    }
    return 0;
}

static int test_max_orders()
{
    int fail = 0;

    fail |= check_val("MAX_ORDER_QCD", MAX_ORDER_QCD, 4);
    fail |= check_val("MAX_ORDER_QED", MAX_ORDER_QED, 2);

    if (!fail) cout << "PASS test_max_orders\n";
    return fail;
}

static int test_pids()
{
    int fail = 0;

    fail |= check_val("PID_NSP",   PID_NSP,   10101);
    fail |= check_val("PID_NSM",   PID_NSM,   10201);
    fail |= check_val("PID_NSV",   PID_NSV,   10200);
    fail |= check_val("PID_NSP_U", PID_NSP_U, 10102);
    fail |= check_val("PID_NSP_D", PID_NSP_D, 10103);
    fail |= check_val("PID_NSM_U", PID_NSM_U, 10202);
    fail |= check_val("PID_NSM_D", PID_NSM_D, 10203);

    if (!fail) cout << "PASS test_pids\n";
    return fail;
}

int main()
{
    int fail = 0;
    fail |= test_max_orders();
    fail |= test_pids();
    return fail ? EXIT_FAILURE : EXIT_SUCCESS;
}
