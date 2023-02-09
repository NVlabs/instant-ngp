#include <iostream>
#include <vector>
#include <string>
#include <assert.h>
#include "natural_sort.hpp"

#define ADD_TEST(test_method) { \
    std::cout<<"Running test " #test_method "..."<<std::endl; \
    test_method(); \
    std::cout<<"Test " #test_method " is successful."<<std::endl; \
}

#define ASSERT_TRUE(x) { assert(x); }
#define ASSERT_FALSE(x) { assert(!(x)); }
#define ASSERT_EQ(x, y) { assert((x) == (y)); }

void test_compare() {
    ASSERT_FALSE(SI::natural::compare<std::string>("Hello 32", "Hello 023"));
    ASSERT_FALSE(SI::natural::compare<std::string>("Hello 32a", "Hello 32"));
    ASSERT_TRUE(SI::natural::compare<std::string>("Hello 32", "Hello 32a"));
    ASSERT_FALSE(SI::natural::compare<std::string>("Hello 32.1", "Hello 32"));
    ASSERT_TRUE(SI::natural::compare<std::string>("Hello 32", "Hello 32.1"));
    ASSERT_FALSE(SI::natural::compare<std::string>("Hello 32", "Hello 32"));
}

void test_sort_vector() {
    std::vector<std::string> data = {
        "Hello 100",
        "Hello 34",
        "Hello 9",
        "Hello 25",
        "Hello 10",
        "Hello 8",
    };
    std::vector<std::string> want = {
        "Hello 8",
        "Hello 9",
        "Hello 10",
        "Hello 25",
        "Hello 34",
        "Hello 100",
    };
    SI::natural::sort(data);
    ASSERT_EQ(data, want);
}

void test_sort_array() {
    const int SZ = 3;
    std::string data[SZ] = {
        "Hello 100",
        "Hello 25",
        "Hello 34",
    };
    std::string want[SZ] = {
        "Hello 25",
        "Hello 34",
        "Hello 100",
    };

    SI::natural::sort<std::string, SZ>(data);
    for(int i=0; i<SZ; i++) {
        ASSERT_EQ(data[i], want[i]);
    }
}

void test_sort_float() {
    std::vector<std::string> data = {
        "Hello 1",
        "Hello 10",
        "Hello 10.3",
        "Hello 2",
        "Hello 10.23",
        "Hello 10.230",
    };
    std::vector<std::string> want = {
        "Hello 1",
        "Hello 2",
        "Hello 10",
        "Hello 10.23",
        "Hello 10.230",
        "Hello 10.3",
    };

    SI::natural::sort(data);
    ASSERT_EQ(data, want);
}

int main()
{
    ADD_TEST(test_compare);
    ADD_TEST(test_sort_vector);
    ADD_TEST(test_sort_array);
    ADD_TEST(test_sort_float);
    return 0;
}