# 算法基础
## 1 排序
<details>
<summary>手撕快速排序</summary>
错解：

```c++
void qsort(vector<int>& nums, int l, int r) {
    if (l >= r) return;
    int i = l - 1, j = r + 1, pivot = l + r >> 1;
    while(i < j) {
        while(nums[++i] < nums[pivot]);
        while(nums[--j] > nums[pivot]); 
        if(i < j) swap(nums[i], nums[j]);
    }
    qsort(nums, l, j);
    qsort(nums, j + 1, r);
}
```
正解：
```c++
void qsort(vector<int>& nums, int l, int r) {
    if (l >= r) return;
    int i = l - 1, j = r + 1, pivot = nums[l + r >> 1];
    while(i < j) {
        while(nums[++i] < pivot);
        while(nums[--j] > pivot); 
        if(i < j) swap(nums[i], nums[j]);
    }
    qsort(nums, l, j);
    qsort(nums, j + 1, r);
}
```
> 错解假设交换时不改变pivot位置，会导致程序死循环。
</details>

<details>
<summary>快排的时间复杂度？快排是否是稳定排序？</summary>

- 均摊 $O(nlogn)$，最坏 $O(n^2)$。
- 不是稳定排序，改为双关键字排序。

</details>

<details>
<summary>什么是快速选择算法？时间复杂度是多少？</summary>

快速选择算法是快速排序算法的扩展应用，可以在$O(n)$的时间内找到无序数组中的第k大（小）数。

时间复杂度期望为$O(n)$,最坏情况为$O(n^2)$。递归带来的空间复杂度期望为$O(logn)$。

推导：
$$ n + n/2 + n/4 + n/8 + ... < 2n $$

通过堆排序找第k大数的时间复杂度为$O(nlogk)$。
```cpp
// LC 215. 数组中的第K个最大元素
int quick_select(vector<int>& nums, int l, int r, int k) {
    if(l >= r) return l;
    int i = l - 1, j = r + 1, mid = nums[l + r >> 1];
    while(i < j) {
        while(nums[++i] < mid);
        while(nums[--j] > mid);
        if(i < j) swap(nums[i], nums[j]);
    }
    int cnt = r - j;
    if(k > cnt) return quick_select(nums, l, j, k - cnt);
    else return quick_select(nums, j + 1, r, k);
}
int findKthLargest(vector<int>& nums, int k) {
    return nums[quick_select(nums, 0, nums.size() - 1, k)];
}
```
</details>


<details>
<summary>手撕归并排序/求逆序对</summary>

```c++
vector<int> tmp;
void merge_sort(vector<int> &nums, int l, int r) {
    if (l >= r) return;
    int mid = (l + r) >> 1;
    merge_sort(nums, l, mid);
    merge_sort(nums, mid + 1, r);
    int i = l, j = mid + 1, k = l;
    while (i <= mid && j <= r) {
        tmp[k++] = nums[i] < nums[j] ? 
                   nums[i++] : nums[j++];
    }
    while (i <= mid) tmp[k++] = nums[i++];
    while (j <= r) tmp[k++] = nums[j++];
    for (i = l; i <= r; i++) nums[i] = tmp[i];
}
```
扩展：
```cpp
// 剑指 Offer 51. 数组中的逆序对
vector<int> tmp;
int msort(vector<int>& nums, int l, int r) {
    if(l >= r) return 0;
    int mid = l + r >> 1;
    int res = msort(nums, l, mid) + msort(nums, mid + 1, r);
    int i = l, j = mid + 1, k = l;
    while(i <= mid && j <= r) {
        if(nums[i] <= nums[j]) tmp[k++] = nums[i++];
        else {
            res += mid - i + 1; // 注意：这一行！！
            tmp[k++] = nums[j++];
        }
    }
    while(i <= mid) tmp[k++] = nums[i++];
    while(j <= r) tmp[k++] = nums[j++];
    for(i = l; i <= r; i++) nums[i] = tmp[i];
    return res;
}
int reversePairs(vector<int>& nums) {
    tmp.resize(nums.size());
    return msort(nums, 0, nums.size() - 1);
}
```

</details>

## 2 二分
<details>
<summary>两种二分写法？</summary>

（1）待求的下标**及其左侧**均满足某种性质，而右侧则不满足
```c++
int l, r, mid = l + r + 1 >> 1;
while(l < r) {
    if(check(mid)) l = mid;
    else r = mid - 1;
}
```
（2）反之，待求的下标**及其右侧**均满足某种性质，而左侧则不满足
```c++
int l, r, mid = l + r >> 1;
while(l < r) {
    if(check(mid)) r = mid;
    else l = mid + 1;
}
```

</details>

<details>
<summary>什么时候使用二分？</summary>

（1）当序列满足某种**单调**性质。（例外，寻找峰值）

（2）**结果范围很大**，但是可以快速判断结果是否正确。
</details>

<details>
<summary>浮点数二分如何保证结果精度？</summary>

假设需要精确到小数点后 6 位。
```c++
double l, r;
while(r - l > 1e-6) {
    double mid = (l + r) / 2;
    if(check(mid)) r = mid;
    else l = mid;
}
```
</details>

##  3 前缀和/差分
<details>
<summary>为什么前缀和可以节约时间？</summary>

假设有一个数组 

$X = \{x_1, x_2, ..., x_n\}$，

对应的前缀和数组为 

$S = \{s_0, s_1, ..., s_n\}$。

其中，

$s_0 = 0, s_i = s_{i-1} + x_i$。

通过前缀和数组，可以用$O(1)$的时间求出迭代需要$O(n)$时间的片段和

$x_i + x_{i+1} + ... + x_j = s_j - s_{i-1}$。
</details>

<details>
<summary>前缀和为什么需要n+1的空间？</summary>

否则在求$[0, i]$的片段和时，$s_{i-1}$会下标越界。
</details>


<details>
<summary>二维前缀和如何计算矩形和？</summary>

预处理二维前缀和 $S$，其中，$S_{i, j}$ 表示第$i$行$j$列格子左上部分所有元素的和。

则以$(x1, y1)$为左上角，$(x2, y2)$为右下角的子矩阵的和为：

$S_{x_2, y_2} - S_{x_1-1,y_2} - S_{x_2, y_1 - 1} + S_{x_1 - 1, y_1 - 1}$。

> 用到左上角$(x1, y1)$的都要$-1$，用到右下角$(x2, y2)$的都不要$-1$。
</details>


<details>
<summary>前缀和与差分的联系？</summary>

前缀和与差分互为**逆运算**，对原数组的差分数组求前缀和会得到原数组。
</details>

<details>
<summary>差分数组用于什么场景？</summary>

需要对数组大量进行片段加减，并最终求原数组的情况，可以先通过$O(n)$的时间构造差分数组，执行若干次$O(1)$的修改操作，最后通过$O(n)$的时间还原数组。

(1) 差分数组的修改操作

假设需要对原数组$[l, r]$执行$+a$的操作，对应差分数组有：
$$ b[l] = b[l] + a, b[r + 1] = b[r + 1] - a$$
因为用到了$r+1$的下标，所以差分数组比原数组在末尾多一个数。

(2)差分数组的构造

初始化差分数组为全0，之后对原数组中每个位置$a_i$执行在$[i, i]$j加$a_i$的操作即可。
</details>

<details>
<summary>二维差分的修改操作如何更新差分数组？</summary>

类似二维前缀和，对以$(x1, y1)$为左上角，$(x2, y2)$为右下角的子矩阵批量$+a$，对应差分数组有：
$$ b[x1, y1] + a, b[x1, y2 + 1] - a, b[x2 + 1, y1] - a, b[x2 + 1, y2 + 1] += a $$
> 对应前缀和，用到左上角$(x1, y1)$的都不要$+1$，用到右下角$(x2, y2)$的都要$+1$。
</details>

## 4 位运算
<details>
<summary>如何找到n的二进制表示的第 k 位（从低到高）？</summary>

n >> k & 1
</details>

<details>
<summary>什么是 lowbit 操作？</summary>

返回 n 的二进制表示的最后一位 1 的位置（返回$000010000$）
$$ lowbit(x) = x \& (-x) $$
原理，x & -x = x & (~x + 1)，
(~x + 1) 在最后一位1以后的位置与x相同，在最后一位1之前的位置与x均相反。-x为x的补码(~x + 1)表示

应用：  
（1）树状数组  
（2）统计n的二进制表示的1的个数  
</details>

## 5 离散化/区间合并
<details>
<summary>离散化的应用场景？</summary>

序列的个数比较少，但是值域很大，即具有“稀疏性”，并且不关心相对顺序。离散化就是将序列中的每个数映射到$[0, n]$，即对数组进行**排序**，排完序对应数组下标即为$[0, n]$。

注意：离散化之后，序列就不能变了。

应用：美团笔试题 [761.格子染色](https://www.acwing.com/problem/content/description/761/)
```cpp
// ACW 761. 格子染色
#include<iostream>
#include<vector>
#include<array>
#include<algorithm>
using namespace std;

vector<array<int, 3>> row, col, raw_row, raw_col;
long long res = 0;
void merge(vector<array<int, 3>>& a, vector<array<int, 3>>& b){
    if(a.empty()) return;
    sort(a.begin(), a.end());
    int id = a[0][0], l = a[0][1], r = a[0][2];
    for(int i = 1; i < a.size(); i++) {
        if(a[i][0] != id || a[i][1] > r) {
            res += r - l + 1, b.push_back({id, l, r});
            id = a[i][0], l = a[i][1], r = a[i][2];
        } else r = max(r, a[i][2]);
    }
    res += r - l + 1, b.push_back({id, l, r});
}
void cross(vector<array<int, 3>>& a, vector<array<int, 3>>& b){
    for(auto aa : a) for(auto bb : b) {
        if(aa[1] <= bb[0] && bb[0] <= aa[2] && bb[1] <= aa[0] && aa[0] <= bb[2]) 
            res --;
    }
}
int main(){
    int n;
    cin >> n;
    for(int i = 0; i < n; i++) {
        int x1, y1, x2, y2;
        cin >> x1 >> y1 >> x2 >> y2;
        if(x1 == x2) raw_row.push_back({x1, min(y1, y2), max(y1, y2)});
        if(y1 == y2) raw_col.push_back({y1, min(x1, x2), max(x1, x2)});
    }
    merge(raw_col, col), merge(raw_row, row);
    cross(row, col);
    cout << res;
    return 0;
}
```
</details>


<details>
<summary>离散化时，原数组有重复元素如何解决？</summary>

对原数组去重。

例如，对数组 nums 去重操作：
```cpp
sort(nums.begin(), nums.end());
nums.erase(unique(nums.begin(), nums.end()), nums.end());
```
</details>
<details>
<summary>离散化时，如何快速找到x对应下标？</summary>

二分，例如在离散化（排完序）后到数组中二分出第一个>=x的位置：
```cpp
int l = 0, r = n;
while(l < r) {
    int mid = l + r >> 1;
    if(nums[mid] >= x) r = mid;
    else l = mid + 1;
}
```
</details>

<details>
<summary>如何手动实现unique函数？</summary>

unique()对升序数组去重,返回去重后数组的尾指针。（注意后半段并非重复元素，没有意义）

实现：两个指针i，j从前往后遍历数组，保持[0, j]中没有重复元素，i往后移动对过程中，若nums[i] != nums [i - 1], nums[++j] = nums[i]。
```cpp
vector<int>::iterator unique(vector<int> &a) {
    int j = 0;
    for(int i = 0; i < a.size(); i++) {
        if(i == 0 || nums[i] != nums[i - 1]){
            nums[++j] = nums[i];
        }
    }
    return a.begin() + j;
}
```
</details>

## 6 链表

<details>
<summary>链表有哪些实现方式？链表的应用场景有哪些？</summary>

实现方式：

（1）结构体+指针：常见于面试题（但是缓存不友好）  
（2）数组模拟：常见于ACM/笔试题  
（3）STL List：API 类似 STL Deque

应用场景：

（1）单链表：以**邻接表**的形式表示树或图，也叫**链式前向星**。  
（2）双链表：**优化**插入删除操作。

场景（2）举例：
```cpp
// LC 2289. 使数组按非递减顺序排列 时间复杂度O(n), 删除操作最多执行n次
int totalSteps(vector<int>& _nums) {
    list<int> nums(_nums.begin(), _nums.end());
    vector<list<int>::iterator> del, tmp;
    for(auto p = nums.begin(); next(p) != nums.end(); ++p) 
        if(*p > *next(p)) del.push_back(next(p));
    reverse(del.begin(), del.end()); // ！必须从后往前删，不然无法判断 p 是否还存在
    int res = 0;
    while(!del.empty()){
        for(auto p : del) {
            if(p == nums.begin() || next(p) == nums.end()) {
                nums.erase(p);
            }else{
                auto pre = prev(p), nxt = next(p);
                nums.erase(p);
                if((tmp.empty() || tmp.back() != nxt) && *pre > *nxt) 
                    tmp.push_back(nxt); // ！需要判重
            }
        }
        del.clear(), del = tmp, tmp.clear(), res++;;
    }
    return res;
}
```
</details>

<details>
<summary>如何加速“结构体+指针”类型的链表？</summary>

可以设计一个内存池`array<T, N> pool`，预先开足够的空间，避免每次执行`new`操作。 $N$为提前开辟的节点数，用`offer()`代替`new()`。
```cpp
template <typename T, std::size_t N = 50000> 
struct memory_pool {
  std::array<T, N> pool = {};
  std::size_t ptr;
  memory_pool() : pool{}, ptr(0) {}
  T *offer() { return &pool[ptr++]; }
};
```
</details>

<details>
<summary>什么是跳表？如何实现？各个操作的时间复杂度是多少？</summary>

跳表是在$O(logn)$时间内完成**增加**、**删除**、**搜索**操作的数据结构。跳表相比于树堆与红黑树，其功能与性能相当，并且跳表的代码长度相较下更短，其设计思想与链表相似。

平时很少会用到，因为C++ STL库中底层为平衡树的`set`也能实现如上操作，同样能维护内部数据有序。

```cpp
// LC 1206. 设计跳表
template <typename T, std::size_t N = 50000> 
struct memory_pool {
    std::array<T, N> pool = {};
    std::size_t ptr;
    memory_pool() : pool{}, ptr(0) {}
    T *offer() { return &pool[ptr++]; }
};

struct node {
    node *right, *down;
    int val;
};

class Skiplist {
private:
    node* head;
    memory_pool<node> pool;
    node* make_node(node* right, node* down, int val) {
        auto p = pool.offer();
        p->right = right, p->down = down, p->val = val;
        return p;
    }

public:
    Skiplist() : pool() {
        head = make_node(nullptr, nullptr, -1);
    }
        
    vector<node*> path(int target) {
        vector<node*> res;
        auto p = head;
        while(p != nullptr) {
            while(p->right != nullptr && p->right->val < target) p = p->right;
            res.push_back(p);
            p = p->down;
        }
        return res;
    }

    bool search(int target) {
        auto p = path(target).back();
        if(p->right != nullptr && p->right->val == target) return true;
        return false;
    }
    
    node* insert(node* p, node* last, int num){
        // 在 p 后面增加一个节点
        p->right = make_node(p->right, last, num);
        return p->right;
    }

    void add(int num) {
        auto down_path = path(num);
        auto p = down_path.back();
        node* last = nullptr;
        bool first = true;
        while(first || rand() % 2 == 1) {
            first = false;
            if(down_path.empty()) { // 增加一层
                head = make_node(make_node(nullptr, last, num), head, -1);
                last = head->right;
            }else { // 从底层往上插
                p = down_path.back();
                last = insert(p, last, num);
                down_path.pop_back();
            }
        }
    }
    
    bool erase(int num) {
        bool flag = false;
        auto down_path = path(num);
        for(auto p : down_path) {
            if(p->right && p->right->val == num) {
                p->right = p->right->right;
                flag = true;
            }
        }
        return flag;
    }
};
```

使用STL multiset 实现：
```cpp
class Skiplist {
private:
    multiset<int> st;

public:
    Skiplist(){}
    
    bool search(int target) {
        return st.count(target);
    }

    void add(int num) {
        st.insert(num);
    }
    
    bool erase(int num) {
        if(!search(num)) return false;
        st.erase(st.find(num)); // 传值会删除所有等于 num 的元素
        return true;
    }
};
```
</details>

## 7 单调栈/队列
<details>
<summary>单调栈应用的场景有哪些？</summary>

在一个序列中，快速找到某个数左边（或右边）**最近的**满足某个性质的数。（单次操作时间复杂度，暴力$O(n)$, 单调栈$O(1)$）

举例，LC 2289的另一种解法
```cpp
// LC 2289. 使数组按非递减顺序排列 时间复杂度O(n)
int totalSteps(vector<int>& nums) {
    stack<array<int, 2>> stk;
    int res = 0;
    for(int i = nums.size() - 1; i >= 0; i--) {
        int m= 0;
        while(!stk.empty() && stk.top()[0] < nums[i]) { 
            // 找到右侧第一个>=nums[i]的位置
            m = max(m + 1, stk.top()[1]); // !重点
            stk.pop();
        }
        stk.push({nums[i], m}), res = max(res, m);
    }
    return res;
}
```
</details>

<details>
<summary>单调队列应用的场景有哪些？</summary>

解决滑动窗口相关问题。
</details>

## 8 字符串

<details>
<summary>什么是KMP算法？原理是怎样的？前缀函数(next数组)如何构建？</summary>

[B站介绍视频](https://www.bilibili.com/video/BV18k4y1m7Ar?p=1&vd_source=793117e7c9233027da3ab9fe378f9bca)

想要解决的问题：查找字符串p在字符串s中第一次出现的位置。时间复杂度$O(m + n)$,m和n为p和s的长度。

原理：构造前缀函数(next数组)，即在j处发生不匹配时，应该跳转到next[j]继续执行匹配。
```cpp
// LC 28. 实现 strStr()
int strStr(string s, string p) {
    int n = s.size(), m = p.size();
    s = ' ' + s, p = ' ' + p;
    vector<int> next(m + 1, 0);
    // 构建 next 数组
    for(int i = 2, j = 0; i <= m; i++) {
        // j 向前跳转到 p[0, i - 1] 中下一个前后缀相同的位置
        // 判断 p[j+1] 是否等于 p[i]
        while(j != 0 && p[i] != p[j + 1]) j = next[j];
        if(p[i] == p[j + 1]) j++;
        next[i] = j;
    }
    for(int i = 1, j = 0; i <= n; i++){
        while(j != 0 && s[i] != p[j + 1]) j = next[j];
        if(s[i] == p[j + 1]) j++;
        if(j == m) return i - m; // 匹配到结尾
    }
    return -1;
}
```
</details>

<details>
<summary>什么是字典树？有哪些应用场景？有哪些实现方法？</summary>

字典树又叫做前缀树、Trie树，是一种高效存储字符串的数据结构。

应用场景：需要快速**动态查询**某个某个字符串出现的次数或是否为某个字符串的**前缀**。
需要支持**插入**和**查询**两个操作，时间复杂度为$O(k)$， k为字符串长度。

实现方法：

(1) **静态模拟**，通过静态数组(结构体)模拟，需要提前开辟足够的空间。
```cpp
// LC 208. 实现 Trie (前缀树)
class Trie {
private:
    vector<int> exist;
    vector<array<int, 26>> next;
    int idx = 1;
public:
    Trie() {
        exist.resize(100010), next.resize(100010);
    }
    
    void insert(string word) {
        int p = 0;
        for(auto c : word) {
            if(next[p][c - 'a']) p = next[p][c - 'a'];
            else next[p][c - 'a'] = idx, p = idx, idx++;
        }
        exist[p] = true;
    }
    
    bool search(string word) {
        int p = 0;
        for(auto c : word) {
            if(next[p][c - 'a']) p = next[p][c - 'a'];
            else return false;
        }
        return exist[p];
    }
    
    bool startsWith(string prefix) {
        int p = 0;
        for(auto c : prefix) {
            if(next[p][c - 'a']) p = next[p][c - 'a'];
            else return false;
        }
        return true;
    }
};
```
(2) **动态维护**，每个节点为一个结构体，通过 new 动态增加新的树节点。
```cpp
// LC 208. 实现 Trie (前缀树)
class Trie {
private:
    array<Trie*, 26> next;
    bool exist;
public:
    Trie():next{}, exist(false){} // 注意 array 的初始化是 {}
    
    void insert(string word) {
        auto p = this;
        for(auto c : word) {
            if(p->next[c - 'a']) p = p->next[c - 'a'];
            else {
                auto node = new Trie();
                p = p->next[c - 'a'] = node;
            }
        }
        p->exist = true;
    }
    
    bool search(string word) {
        auto p = this;
        for(auto c : word) {
            if(p->next[c - 'a']) p = p->next[c - 'a'];
            else return false;
        }
        return p->exist;
    }
    
    bool startsWith(string prefix) {
        auto p = this;
        for(auto c : prefix) {
            if(p->next[c - 'a']) p = p->next[c - 'a'];
            else return false;
        }
        return true;
    }
};
```

使用内存池，动态改静态
```cpp
template<typename T, size_t N = 50000>
struct memory_pool{
    array<T, N> pool;
    size_t idx;
    memory_pool() : pool{}, idx(0) {}
    T* offer(){return &pool[idx++];}
};

struct node{
    array<node*, 26> next;
    bool exist;
};

class Trie {
private:
    memory_pool<node> pool;
    node* root;

public:
    Trie():pool(){root = pool.offer();} // 注意 array 的初始化是 {}
    
    void insert(string word) {
        auto p = root;
        for(auto c : word) {
            if(p->next[c - 'a']) p = p->next[c - 'a'];
            else {
                auto node = pool.offer();
                p = p->next[c - 'a'] = node;
            }
        }
        p->exist = true;
    }
    
    bool search(string word) {
        auto p = root;
        for(auto c : word) {
            if(p->next[c - 'a']) p = p->next[c - 'a'];
            else return false;
        }
        return p->exist;
    }
    
    bool startsWith(string prefix) {
        auto p = root;
        for(auto c : prefix) {
            if(p->next[c - 'a']) p = p->next[c - 'a'];
            else return false;
        }
        return true;
    }
};
```
</details>


## 9 并查集
<details>
<summary>并和查分别指什么？它的时间复杂度为？它有哪些经典的应用？</summary>

- 并：快速合并两个集合；   
- 查：查询元素属于哪个集合，还可以查询集合的大小，元素到根结点距离等。

时间复杂度：使用路径压缩、按秩合并后可达到 $O(\alpha(n))$, $\alpha(n)$ 为阿克曼函数的反函数,可以认为是一个很小的常数，近乎$O(1)$。

应用：常见的应用有**带权并查集**，最小生成树算法中的**Kruskal**和最近公共祖先中的**Tarjan**算法。
[OI-Wiki](https://oi-wiki.org/topic/dsu-app/)上整理了一些并查集在图论中的应用。

</details>

<details>
<summary>并查集的优化方法有哪些？</summary>

有**路径压缩**和**按秩合并**两个方法，在算法竞赛的实际代码中，即便不使用按秩合并，代码也往往能够在规定时间内完成任务。在 Tarjan 的论文中，证明了不使用按秩合并、只使用路径压缩的最坏时间复杂度是$O(logn)$。在姚期智的论文中，证明了不使用按秩合并、只使用路径压缩，在平均情况下，时间复杂度依然是 $O(\alpha(n))$。
</details>


<details>
<summary>并查集的find和union操作如何实现？</summary>

- `find` 操作：
```cpp
int find(int i){
    if(p[i] != i) p[i] = find(p[i]);
    return p[i];
}
```
- `union` 操作：(注意：不要定义成`union`，c++中`union`是联合体，类似`struct`是结构体)

```cpp
void Union(int i, int j) {
    p[find(i)] = p[find(j)];
}
```
例如，
```cpp
// LC 547. 省份数量
class Solution {
public:
    vector<int> p;
    int find(int i){
        if(p[i] != i) p[i] = find(p[i]);
        return p[i];
    }
    void Union(int i, int j) {p[find(i)] = p[find(j)];}
    int findCircleNum(vector<vector<int>>& isConnected) {
        int n = isConnected.size(), res = n;
        p.resize(n);
        for(int i = 0; i < n; i++) p[i] = i;
        for(int i = 0; i < n; i++) for(int j = 0; j < n; j++) {
            if(isConnected[i][j] != 0 && find(i) != find(j)) {
                res--, Union(i, j);
            }
        }
        return res;
    }
};
```
</details>


## 10 堆
<details>
<summary>堆有哪些操作？时间复杂度是多少？</summary>

堆有插入、弹出堆顶和删除指定元素的操作。插入、删除和弹出均为$O(logn)$，取堆顶元素为$O(1)$。
</details>

<details>
<summary>堆如何实现？如何使用up和down操作实现各个操作？</summary>

堆可以通过一个静态数组模拟（从1开始），第`i`位置的左右儿子分别是`2i`和`2i+1`。

- 插入：up(idx++)
- 取堆顶：heap[1]
- 弹出堆顶：heap[1] = heap[idx--], down(1)
- 删除堆中的某个元素：heap[k] = heap[idx--], down(k), up(k)

```cpp
// LC 703. 数据流中的第 K 大元素
class KthLargest {
    // 如果已经有k个元素比当前元素大了，那当前元素必然不是答案
    vector<int> heap;
    int idx, k;
    void up(int x) {
        if(x != 1 && heap[x / 2] > heap[x]) {
            swap(heap[x / 2], heap[x]), up(x / 2);
        }
    }
    void down(int x) {
        if(x * 2 < idx && heap[x * 2] < heap[x]) {
            swap(heap[x * 2], heap[x]), down(x * 2);
        } 
        if(x * 2 + 1 < idx && heap[x * 2 + 1] < heap[x]) {
            swap(heap[x * 2 + 1], heap[x]), down(x * 2 + 1);
        }
    }
public:
    KthLargest(int k, vector<int>& nums) {
        heap.resize(k + 1);
        this->k = k;
        idx = 1;
        for(auto x : nums) {
            if(idx > k) {
                if(x > heap[1]) heap[1] = x, down(1);
            } else {
                heap[idx] = x, up(idx), idx++;
            }
        }
    }
    int add(int val) {
        if(idx > k) {
            if(val > heap[1]) heap[1] = val, down(1);
        } else {
            heap[idx] = val, up(idx), idx++;
        }
        return heap[1];
    }
};
```
</details>

## 11 哈希表
<details>
<summary>什么是哈希表？什么是哈希函数？哈希表和离散化有什么区别？</summary>

- 哈希表又称散列表，一种以 "key-value" 形式存储数据的数据结构。

- 哈希函数，是将一个较大值域的输入($>10^9$)，映射到一个较小的值域输出($<10^5$)。对于数值来说，一般采用**取模**的方式。

- 离散化是一种特殊的哈希函数，相比普通的哈希函数，它还具有**保序**的特点。
</details>

<details>
<summary>当哈希函数出现碰撞时如何解决？</summary>

通常有**拉链法**和**开放寻址法**。
- 拉链法（链表实现）
(1) unordered_set
```cpp
// LC 705. 设计哈希集合
template<typename T, const size_t N = 10010>
struct memory_pool{
    array<T, N> pool;
    size_t ptr;
    memory_pool():pool{}, ptr(0){}
    T* offer(){return &pool[ptr++];}
};
struct node{
    int val;
    node *next;
    node():val(-1), next(nullptr){}
};
const int MOD = 769;
class MyHashSet {
    array<node*, 769> hash;
    memory_pool<node> pool;
    node* make_node(int val, node* next) {
        auto p = pool.offer();
        p->val = val, p->next = next;
        return p;
    }
public:
    MyHashSet() :hash{}{}
    
    void add(int key) {
        if(contains(key)) return;
        int idx = key % MOD;
        auto p = hash[idx];
        if(!p){
            hash[idx] = make_node(key, nullptr);
            return;
        }
        while(p->next) p = p->next;
        p->next = make_node(key, nullptr);
    }
    
    void remove(int key) {
        int idx = key % MOD;
        auto p = hash[idx];
        if(!p) return;
        if(p->val == key) hash[idx] = p->next;
        else {
            auto q = p->next;
            while(q && q->val != key) p = q, q = q->next;
            if(q && q->val == key) p->next = q->next;
        }
    }
    
    bool contains(int key) {
        int idx = key % MOD;
        auto p = hash[idx];
        while(p && p->val != key) p = p->next;
        if(!p || p->val != key) return false;
        else return true;
    }
};
```
(2) unordered_map
```cpp
// LC 706. 设计哈希映射
template<typename T, const size_t N = 10010>
struct memory_pool{
    array<T, N> pool;
    size_t ptr;
    memory_pool():pool{}, ptr(0){}
    T* offer(){return &pool[ptr++];}
};
struct node{
    int key, val;
    node* next;
};
const int MOD = 769;
class MyHashMap {
    array<node*, 769> hash;
    memory_pool<node> pool;
    node* make_node(int key, int val){
        auto p = pool.offer();
        p->key = key, p->val = val;
        return p;
    }
public:
    MyHashMap() :hash{}{}
    
    void put(int key, int value) {
        int idx = key % MOD;
        if(hash[idx] == nullptr) {
            hash[idx] = make_node(key, value);
        }else if(hash[idx]->key == key){
            hash[idx]->val = value;
        }else{
            auto p = hash[idx], q = p->next;
            while(q && q->key != key) p = q, q = q->next;
            if(q && q->key == key) q->val = value;
            else p->next = make_node(key, value);
        }
    }
    
    int get(int key) {
        int idx = key % MOD;
        auto p = hash[idx];
        while(p && p->key != key) p = p->next;
        if(p) return p->val;
        else return -1;
    }
    
    void remove(int key) {
        if(get(key) == -1) return;
        int idx = key % MOD;
        if(hash[idx]->key == key) {
            hash[idx] = hash[idx]->next;
        }else {
            auto p = hash[idx], q = p->next;
            while(q && q->key != key) p = q, q = q->next;
            p->next = q->next;
        }
    }
};
```
</details>


## 12 图论

<details>
<summary>图有哪些存储方式？什么是图的出度和入度？</summary>

假设，`N` 为节点数， `M` 为边数
- 邻接矩阵 -> 适合存储稠密图
```cpp
int G[N][N]; // array<array<int, N>, N> G;
```
- 邻接表   -> 适合存储稀疏图
```cpp
array<vector<int>, N> G; // 权重均为 1 
struct node{
    int end, val;
};
array<vector<node>, N> G; // 带权重
```
- 边集
```cpp
struct node{
    int start, end, val;
}
array<node, M> edges;
```

- 出度：起点为某个点的边数
- 入度：终点为某个点的边数
</details>

<details>
<summary>什么是拓扑图？什么是拓扑序列？如何求拓扑序列？</summary>

- 拓扑图：有向无环图
- 拓扑序列：按某个序列将图中所有节点排序，满足所有边均从前指向后
- 如何求拓扑序列？等价于如何判断一个图是拓扑图？等价于一个有向图有否存在环？
    - 维护每个点的入度，每次找入度为0的点加入序列
    - 使用**队列**维护所有入度已经是0的点
    - 队列可以用数组模拟，**入队顺序**构成拓扑序列
    ```cpp
    // LC 210. 课程表 II
    vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
        vector<int> in(numCourses); // 入度
        vector<vector<int>> G(numCourses); // 邻接表
        for(auto x : prerequisites) G[x[1]].push_back(x[0]), in[x[0]] ++;
        vector<int> q;
        for(int i = 0; i < numCourses; i++) if(in[i] == 0) q.push_back(i);
        int idx = 0;
        while(idx < q.size()) {
            for(auto x : G[q[idx++]]) {
                if(--in[x] == 0) 
                    q.push_back(x);
            }
        }
        return idx == numCourses ? q : vector<int>();
    }
    ```

</details>

<details>
<summary>常见的最短路问题有哪些？分别对应什么解法？如何判断稠密图和稀疏图？</summary>

常见的最短路问题：
- 单源最短路  
    - 不存在负权边  
        - 稠密图：朴素 Dijkstra -> $O(n^2)$  
        - 稀疏图：堆优化 Dijkstra -> $O(mlogn)$  
    - 存在负权边  
        - Bellman-Ford -> $O(mn)$  
        - SPFA -> $O(m)$, 最坏情况$O(mn)$  
- 多源最短路  
    - Floyd -> $O(n^3)$  
如何判断稠密图和稀疏图？
- 稠密图：$m \approx n^2$
- 稀疏图：$m \approx n$

例如，LC 743. 网络延迟时间， N <= 100, M <= 6000
$m \approx n^2$,为稠密图且无负权边，适合解法：朴素 Dijkstra > SPFA > 堆优化 Dijkstra > Floyd > Bellman-Ford 
```cpp
// LC 743. 网络延迟时间
// 朴素 Dijkstra 84 ms
int networkDelayTime(vector<vector<int>>& times, int n, int k) {
    vector<vector<int>>G(n, vector<int>(n, INT_MAX));
    vector<int> st(n, false), d(n, INT_MAX);
    for(auto &x:times) G[x[0] - 1][x[1] - 1] = x[2];
    d[k - 1] = 0;
    for(int m = 0; m < n; m++) {
        int u = -1;
        for(int i = 0; i < n; i++) {
            if(!st[i] && (u == -1 || d[i] < d[u])) u = i;
        }
        if(d[u] == INT_MAX) break;
        st[u] = true;
        for(int v = 0; v < n; v++) {
            if(!st[v] && G[u][v] != INT_MAX)
                d[v] = min(d[v], d[u] + G[u][v]);
        }
    }
    int res = 0;
    for(auto x : d) res = max(res, x);
    return res == INT_MAX ? -1 : res;
}

// SPFA 84 ms
int networkDelayTime(vector<vector<int>>& times, int n, int k) {
    vector<vector<pii>> G(n);
    vector<int> st(n, false), d(n, INT_MAX);
    for(auto &x : times) G[x[0] - 1].push_back({x[1] - 1, x[2]});
    queue<int> q;
    q.push(k - 1);
    d[k - 1] = 0, st[k - 1] = true;
    while(!q.empty()){
        auto t = q.front();
        q.pop();
        st[t] = false;
        for(auto x : G[t]) 
            if(d[t] + x.second < d[x.first]) {
                d[x.first] = d[t] + x.second;
                if(st[x.first] == false) q.push(x.first), st[x.first] = true;
            }
    }
    int res = 0;
    for(auto x : d) res = max(res, x);
    return res == INT_MAX ? -1 : res;
}

// 堆优化 Dijkstra 96 ms
int networkDelayTime(vector<vector<int>>& times, int n, int k) {
    vector<vector<pii>> G(n);
    vector<int> st(n, false), dis(n, INT_MAX);
    for(auto &x : times) G[x[0] - 1].push_back({x[1] - 1, x[2]}); // {j, d[i,j]}
    priority_queue<pii, vector<pii>, greater<pii>> q; // {dis[idx], idx}
    q.push({0, k - 1});
    dis[k - 1] = 0;
    while(!q.empty()) {
        auto t = q.top();
        q.pop();
        if(st[t.second] == false) {
            st[t.second] = true;
            for(auto x : G[t.second]) {
                if(dis[t.second] + x.second < dis[x.first]){
                    dis[x.first] = dis[t.second] + x.second;
                    q.push({dis[x.first], x.first});
                }
            } 
        }
    }
    int res = 0;
    for(auto x : dis) res = max(res, x);
    return res == INT_MAX ? -1 : res;
}

// Floyd 192 ms
int networkDelayTime(vector<vector<int>>& times, int n, int k) {
    vector<vector<int>> G(n, vector<int>(n, 500));
    for(auto &x : times) G[x[0] - 1][x[1] - 1] = x[2];
    for(int i = 0; i < n; i++) G[i][i] = 0;
    for(int k = 0; k < n; k++)
        for(int i = 0; i < n; i++)
            for(int j = 0; j < n; j++)
                G[i][j] = min(G[i][j], G[i][k] + G[k][j]);
    int res = 0;
    for(auto x : G[k - 1]) res = max(res, x);
    return res == 500 ? -1 : res;
}

// Bellman Ford 200 ms
int networkDelayTime(vector<vector<int>>& times, int n, int k) {
    vector<int> d(n, INT_MAX);
    d[k - 1] = 0;
    while(n--)
        for(auto &x : times) 
            if(d[x[0] - 1] != INT_MAX)
                d[x[1] - 1] = min(d[x[1] - 1], d[x[0] - 1] + x[2]);
    int res = 0;
    for(auto x : d) res = max(res, x);
    return res == INT_MAX ? -1 : res;
}
```
</details>

<details>
<summary>如何判断是否存在负权回路？什么时候使用Bellman Ford算法？</summary>

（1）如何判断是否存在负权回路？
- SPFA：更新dis数组时同时维护cnt数组，若cnt中出现大于n的值，则存在负权回路。
    - 注意：初始时需要将**所有点**入队！
- Bellman Ford：外层循环执行超过n次时，dis数组仍然更新时，存在负权回路。

（2）什么时候使用Bellman Ford算法？
- 有**边数限制**时使用Bellman Ford算法

```cpp
// LC 743. 网络延迟时间 增加cnt数组
int networkDelayTime(vector<vector<int>>& times, int n, int k) {
    vector<vector<pii>> G(n);
    vector<int> st(n, true), d(n, INT_MAX), cnt(n, 0); // cnt 判断负环
    for(auto &x : times) G[x[0] - 1].push_back({x[1] - 1, x[2]});
    queue<int> q;
    for(int i = 0; i < n; i ++) q.push(i);
    d[k - 1] = 0;
    while(!q.empty()){
        auto t = q.front();
        q.pop();
        st[t] = false;
        if(d[t] == INT_MAX) continue;
        for(auto x : G[t]) 
            if(d[t] + x.second < d[x.first]) {
                d[x.first] = d[t] + x.second;
                if(st[x.first] == false) 
                    q.push(x.first), st[x.first] = true;
                cnt[x.first] = cnt[t] + 1;
                if(cnt[x.first] > n) return -1; // 判断是否存在负环
            }
    }
    int res = 0;
    for(auto x : d) res = max(res, x);
    return res == INT_MAX ? -1 : res;
}
```
</details>

<details>
<summary>什么是最小生成树？通常有哪些解法，适用于哪些场景？</summary>

**无向连通图**的最小生成树（Minimum Spanning Tree，MST）为**边权和最小**的生成**树**。简单来说，就是选哪些边之后，所有节点联通，这些边的权值和最小。

- 朴素 Prim 算法: 适合稠密图，时间复杂度$O(n^2)$
- Kruskal算法: 适合稀疏图，时间复杂度$O(mlogm)$

```cpp
// LC 1584. 连接所有点的最小费用

// Prim 算法 O(n^2) 108 ms
int minCostConnectPoints(vector<vector<int>>& p) {
    vector<vector<int>> G(p.size(), vector<int>(p.size(), INT_MAX));
    vector<int> st(p.size(), false), d(p.size(), INT_MAX);
    for(int i = 0; i < p.size(); i++) {
        for(int j = i + 1; j < p.size(); j++) {
            int d = abs(p[i][0] - p[j][0]) + abs(p[i][1] - p[j][1]);
            G[i][j] = d, G[j][i] =  d;
        }
    }
    int res = 0;
    d[0] = 0;
    for(int i = 0; i < p.size(); i++) {
        int t = -1;
        for(int j = 0; j < p.size(); j++) {
            if(st[j] == false && (t == -1 || d[j] < d[t]))
                t = j;
        }
        res += d[t], st[t] = true;
        for(int i = 0; i < p.size(); i++) d[i] = min(d[i], G[t][i]);
    }
    return res;
}

// Kruskal 算法 O(mlogm) = O(n^2logn) 560 ms
vector<int> f;
int find(int x) {
    if(f[x] != x) f[x] = find(f[x]);
    return f[x];
}
int minCostConnectPoints(vector<vector<int>>& p) {
    vector<array<int, 3>> edges;
    f.resize(p.size());
    for(int i = 0; i < p.size(); i++) {
        for(int j = i + 1; j < p.size(); j++) {
            int d = abs(p[i][0] - p[j][0]) + abs(p[i][1] - p[j][1]);
            edges.push_back({d, i, j});
        }
    }
    sort(edges.begin(), edges.end());
    for(int i = 0; i < p.size(); i++) f[i] = i;
    int res = 0;
    for(auto &x : edges){
        x[1] = find(x[1]), x[2] = find(x[2]);
        if(x[1] != x[2]) {
            res += x[0];
            f[x[1]] = x[2];
        }
    }
    return res;
}
```

</details>

<details>
<summary>如何判断一个图是不是二分图？</summary>

利用**染色法**判断一个图是不是二分图：
**二分图**等价于**可以二染色**等价于**图中不存在奇数环**。
```cpp
// LC 785. 判断二分图 DFS 二染色法
vector<int> color;    
bool dfs(vector<vector<int>>& graph, int u) {
    for(auto v : graph[u]){
        if(color[v] == color[u]) return false;
        if(color[v] == -1) {
            color[v] = 1 - color[u];
            if(dfs(graph, v) == false) return false;
        }
    }
    return true;
}
bool isBipartite(vector<vector<int>>& graph) {
    color.resize(graph.size(), -1);
    for(int i = 0; i < graph.size(); i++) {
        if(color[i] == -1) {
            color[i] = 0;
            if(dfs(graph, i) == false) return false;
        }
    }
    return true;
}
```

二分图与二分组合使用：
```cpp
// AWC 257. 关押罪犯
#include<iostream>
#include<vector>
using namespace std;
typedef pair<int, int> pii;

vector<vector<pii>> G;
vector<int> color;

bool dfs(int u, int mid) {
    for(auto v : G[u]) {
        if(v.second <= mid) continue;
        if(color[v.first] == color[u]) return false;
        if(color[v.first] == -1) {
            color[v.first] = 1 - color[u];
            if(dfs(v.first, mid) == false) return false;
        }
    }
    return true;
}

bool check(int mid) {
    color = vector<int>(G.size(), -1);
    for(int i = 0; i < G.size(); i++) {
        if(color[i] == -1) {
            color[i] = 0;
            if(dfs(i, mid) == false) return false;
        }
    }
    return true;
}

int main(){
    int n, m;
    cin >> n >> m;
    G.resize(n + 1);
    for(int i = 0; i < m; i++) {
        int a, b, c;
        cin >> a >> b >> c;
        G[a].push_back({b, c}), G[b].push_back({a, c});
    }
    int l = 0, r = 1e9 + 10;
    while(l < r) {
        int mid = l + r >> 1;
        // check: 怒气值大于mid的边可以构成二分图
        if(check(mid)) r = mid; 
        else l = mid + 1;
    }
    cout << l;
    return 0;
}
```

</details>

<details>
<summary>如何求一个二分图的最大匹配？
</summary>


利用**匈牙利算法**找二分图的最大匹配。若交换后可以增加匹配数就交换，通过dfs实现递归寻找过程。

在目标检测、车道线检测等场景，可以利用匈牙利算法寻找**预测的目标**与**真值**的最大匹配，作为评测指标。
**例如**，预测了`n`条车道线，真值为`m`条车道线，我们可以设定一个阈值（例如两条车道线之间重叠的部分占超过各自的`75%`，重叠表示在同一距离，横向偏差小于 `0.5` 米等）来判断`pred_i`和`gt_j`是否为一对匹配，之后使用匈牙利算法计算这个二分图的最大匹配。

**棋盘匹配**是一个经典的匈牙利算法应用场景。我们可以将相邻的格子分别染色为黑白，**黑色格子和白色格子之间的边构成二分图**，可将问题转化为二分图求最大匹配。
```cpp
// LCP 04. 覆盖 (https://leetcode.cn/problems/broken-board-dominoes/)
int n, m;
vector<int> st, match, G;
int dx[4] = {-1, 0, 1, 0}, dy[4] = {0, -1, 0, 1};
bool find(int x, int y) {
    for(int i = 0; i < 4; i++) {
        int xx = x + dx[i], yy = y + dy[i];
        if(xx < 0 || xx >= n || yy < 0 || yy >= m 
            || G[xx * m + yy] == false || st[xx * m + yy] == true) continue; 
        st[xx * m + yy] = true;
        if(match[xx * m + yy] == -1 ||
                find(match[xx * m + yy] / m, match[xx * m + yy] % m)) {
            match[xx * m + yy] = x * m + y;
            return true;
        }
    }
    return false;
}
int domino(int n, int m, vector<vector<int>>& broken) {
    this->n = n, this->m = m;
    match.resize(n*m, -1), G.resize(n*m, true);
    for(auto &t : broken)  G[t[0] * m + t[1]] = false;
    int res = 0;
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            if((i + j) % 2 == 1 && G[i * m + j] == true) {
                st = vector<int>(n*m, false);
                if(find(i, j)) res ++;
            }
        }
    }
    return res;
}
```
</details>

## 13 数论
<details>
<summary>如何快速判断一个数是否是质数？如何分解质因数？有哪些质数筛法？如何求一个数的约数？</summary>

- 判断质数：（试除法）遍历从2到$\sqrt n$，时间复杂度$O(\sqrt n)$
- 分解质因数：（试除法）遍历从2到$\sqrt n$，直到除为1终止，**最坏**时间复杂度$O(\sqrt n)$
- 求约数：（试除法）遍历从2到$\sqrt n$，时间复杂度$O(\sqrt n)$
- 质数筛：
    - 埃式筛法：时间复杂度$O(nloglogn)$，从小到大筛掉所有**质数**的倍数。
    - 线性筛法：时间复杂度$O(n)$，只会被最小质因数筛去。

> 注意：`sqrt`操作需要替换成 `i < n / i`。
```cpp
// LC 204. 计数质数
// 线性筛
int countPrimes(int n) {
    vector<int> p, st(n, true); // p 动态维护已经找到的质数
    for(int i = 2; i < n; i++) {
        if(st[i] == true) p.push_back(i);
        for(auto x : p) { // 不是只对质数操作！
            if(x * i < n) st[x * i] = false;
            else break; // 注意检查 x*i是否会溢出！
            if(i % x == 0) break;
        }
    }
    return p.size();
}

// 埃式筛
int countPrimes(int n) {
    vector<int> st(n, true);
    int res = 0;
    for(int i = 2; i < n; i++) {
        if(st[i] == true) {
            res ++;
            for(int j = 2; i * j < n; j++) st[i * j] = false;
        }
    }
    return res;
}
```
</details>

<details>
<summary>如何求两个数的最大公约数和最小公倍数？</summary>

欧几里得算法（辗转相除法）原理：$(a, b) = (b, a \mod b)$, `()`表示求最大公因数  

最小公倍数 = a*b / 最大公因数

时间复杂度$O(logn)$
```cpp
// ACW 3642. 最大公约数和最小公倍数
#include<iostream>
using namespace std;
int gcd(int a, int b) {
    return b == 0 ? a : gcd(b, a % b);
}
int main(){
    int a, b;
    cin >> a >> b;
    cout << gcd(a, b) << ' ' << a * b / gcd(a, b) << endl;
    return 0;
}
```
</details>

<details>
<summary>什么是快速幂？</summary>

求解问题：$a^k \mod p = ?$

原理：`k`拆分成若干个2的次幂和， $k = ? 2^0 + ? 2^1 + ... + ?2^{logk}$, `?`处是`k`的二进制表示。  
原问题就表示为$a^k \mod p = a^{? 2^0 + ? 2^1 + ... + ?2^{logk}} \mod p$

$ = (a^{? 2^0} * a^{? 2^1} * ... * a^{?2^{logk}}) \mod p $

```cpp
// LC 50. Pow(x, n) 没有求 mod 的操作
double myPow(double x, int n) {
    if(n == INT_MIN) // 特判 INT_MIN，-INT_MIN会溢出
        if(x == 1.0 || x == -1.0) return 1;
        else return 0;
    if(n < 0) // 特判 负数
        return 1.0 / myPow(x, -n);
    double res = 1.0;
    while(n) {
        if(n & 1) res *= x;
        x *= x;
        n >>= 1;
    }
    return res;
}
```
</details>

<details>
<summary>什么是约数定理？什么是欧拉函数？什么是中国剩余定理？。。。</summary>
不可能考～
</details>

<details>
<summary>test</summary>

test
</details>