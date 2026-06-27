#pragma once
#include <vector>
#include <iostream>
#include <initializer_list>
#include <cassert>
using namespace std;
using ll=long long;
#define sign(x) (x<0?-1:1)
#define sz(x) ((int)size(x))
#define all(x) begin(x),end(x)
#define all1(x) begin(x)+1,end(x)
template <typename T> using vec=vector<T>;
template <typename T> struct vec2:vector<vector<T>> {
    vec2()=default;
    // create sized 2d vector
    vec2(int n, int m, T val=T()):vector<vector<T>>(n,vector<T>(m,val)){}
    // create 2d vector from brace lists
    vec2(initializer_list<vector<T>> rows):vector<vector<T>>(rows){}
    void assign(int n, int m, T val = T()) {this->vector<vector<T>>::assign(n, vector<T>(m, val));}
};
template <typename T> struct vec3:vector<vector<vector<T>>> {vec3()=default;vec3(int n, int m, int k, T val=T()):vector<vector<vector<T>>>(n,vector<vector<T>>(m,vector<T>(k,val))){}void assign(int n, int m, int k, T val = T()) {this->vector<vector<vector<T>>>::assign(n, vector<vector<T>>(m, vector<T>(k, val)));}};
template <typename T> void vprint(T st, T nd) {auto it=st;while (next(it)!=nd){cout<<*it<<' ';it=next(it);}cout<<*it<<'\n';}
template <typename T> bool ckmin(T &a, T b) {return b<a ? a=b, true : false;}
template <typename T> bool ckmax(T &a, T b) {return b>a ? a=b, true : false;}

template <typename T>
inline void read_bytes_count(const vec<uint8_t> &bytes, size_t &pos, T *dest, uint64_t byte_cnt) {
    assert(pos <= bytes.size());
    assert(byte_cnt <= bytes.size() - pos);

    memcpy(dest, bytes.data() + pos, byte_cnt);
    pos += byte_cnt;
}

template <typename T>
inline T read_bytes(const vec<uint8_t> &bytes, size_t &pos) {
    T out;
    read_bytes_count(bytes, pos, &out, sizeof(T));
    return out;
}

template <typename T>
inline T append_bytes_count(vec<uint8_t> &bytes, T *start, uint64_t byte_cnt) {
    size_t old_sz = sz(bytes);
    bytes.resize(old_sz + byte_cnt);
    memcpy(bytes.data() + old_sz, start, byte_cnt);
}

template <typename T>
inline T append_bytes(vec<uint8_t> &bytes, const T &value) {
    append_bytes_count(bytes, &value, sizeof(T));
}
