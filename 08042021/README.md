# Notes

## 1. Pointers

Let `int y;` then `&y` means the memory direction of y.


```c++
char c;
int* pc; 
pc = &c; // error
```

```c++
char c;
char* pc; 
pc = &c // no error
```

- ### Declaration

```c++
int x;
int* px; // declaration
```

- ### Initialization

Initialization of a pointer is when in its declaration, you also assign its value, which is a memory address.

```c++
int* px = &x;
```

- ### Assignment

After a pointer declaration, you can assign a memory address as the pointer value.

```c++
px = &x;
```

- ### Indirection

Indirection is accessing to the value of the variable the pointer is pointing at.

```c++
cout << *px;
```

### Pointers To Pointers

```c++
float y = 2.5;
float *py = &y; // needs to be float because y is float // simple pointer
float **ppy = &py; // mem address of the pointer py, needs to be float because of y
float ***pppy = &ppy; // triple pointer, not so common
cout << ***pppy; // 2.5
```

### Type of Pointers

- **void pointers**

Also called a generic pointer, they do not have a defined value of the variable they will point at, most of the times because the variable type they will point at is unknown. Void Pointers can point to whatever type of value. 

```c++
char c;
int x;
void* pc;

pc = &c;
pc = &x; // all valid
```

- **null pointers**

A null pointer does not point to any direction, does not point to trash.

```c++
char c;
char* p = NULL;
```

- **constant pointers (int \*const)**

```c++
int x = 4;
int* const px = &x; // pointer cannot change its memory address (but x can change), and must be initialized, else error
cout << *px; // 4 (print the value of x through a pointer)
*px = 8; // x or *px can change
cout << *px; // 8
// error px = &y;
```

- **pointers to constants (const \*int)**

```c++
const int x_const = 12;
// error int* p3 = &x_const; 
const int* p3 = &x_const; // no error, also this pointer can point to different const int variables
*p3 = 11 // error bc you're changing x_const or p3
const int y_const = 10;
p3 = &y_const; // p3 can change its target, but its variable type is const int
```

### Pointers To Arrays

The name of an array is a pointer to its first element.

```c++
int arr[3] = {5,7,9};
int *p = arr; // same as p = &arr[0]
cout << *p; // 5
cout << p[0]; // 5
cout << *(p + 1); // 7, bc we are advancing one memory cell
cout << *(p++); // 7
```

### Exercise 1

Code a function with no return value that receives three pointers to vectors (arrays) of integers.The function must add the vectors (v1[1] + v2[1]). For the implementation use:

- 2 arrays initialized with 5 elements each.

- 1 array to store the sum.

- 3 pointers that point to these vectors.

```c++
#include <iostream>
using namespace std;

void func(int* p1, int* p2, int* pr) {

    for (int i = 0; i < 5; i++) {
        pr[i] = p1[i] + p2[i];
    }

    return;
}

int main()
{
    int a1[5] = { 1,2,3,4,5 };
    int a2[5] = { 6,7,8,9,10 };
    int r[5] = { 0 };

    int* pr = r;
    int* p1 = a1;
    int* p2 = a2;

    func(p1, p2, pr);

    for (int i = 0; i < 5; i++) {
        cout << pr[i] << " "; // 7, 9, 11, 13, 15
    }

    for (int i = 0; i < 5; i++) {
        cout << r[i] << " "; // 7, 9, 11, 13, 15
    }
}
```

### Pointers To Arrays of Pointers

It is posible to create arrays of pointers, and pointers that point to these arrays.

```c++
int v_int = 12;
int v_int2 = 3;

int *pt_array[3]; // array of pointers
int  **p_pt_array = pt_array; // pointer to array of pointers

pt_array[0] = &v_int;
pt_array[1] = &v_int2; // same as p_pt_array[1] = &v_int2
```

## 2. Dynamic Memory

The **new** operator in c++ is used to reserve memory in execution time, while the operator **delete** frees this memory.

- Syntaxis

```c++
void* new dataType;
void delete void* block;
void delete [] void* block;
```

- Examples

```c++
int* p;
p = new int;
*p = 10;
cout << *p; // 10
delete p;
```

```c++
int* p;
p = new int [10];
for (int i = 0; i < 10; i++){
    p[i] = i;
    cout << p[i] << "-";
}
delete [] p;
```

C's function `malloc()` is used to reserve memory in execution time, while the function `free()` frees this memory.

- Syntaxis

```c++
void* malloc(size_t size)
void free(void* block)
```

- `malloc()` returns a void* pointer and so we cast it as `(int *)`.

```c++
int *p;
p = (int*)malloc(sizeof(int)); // create a space in dynamic memory (exec mem) that is referenced by p pointer
*p = 45
cout<<*p;
free(p); // param is the name of the pointer that references to mem you want to free
```

```c++
int *p;
p = (int*)malloc(sizeof(int) * 10);
for(int i = 0; i < 10; i++){
    p[i] = i;
    cout << p[i] << "-";
}
free(p);
```

### Exercise 2

Code a function that has no return value, and that it receives two pointers to integers. The function must swap the values of the parameters that it receives. These values must be from user input and stored in dynamic memory. For the implementation, use:

- Dynamic memory

- Pointers

```c++
#include <iostream>
using namespace std;

void func(int *p1, int *p2) {
    
    int aux;

    aux = *p1;
    *p1 = *p2;
    *p2 = aux;

    return;
}

int main()
{
    int* v1 = (int*)malloc(sizeof(int));
    int* v2 = (int*)malloc(sizeof(int));

    cin >> *v1;
    cin >> *v2;

    func(v1, v2);
    cout << "v1: " << *v1 << "\nv2: " << *v2; // swap values of v1 and v2
}
```

## Some Findings

```c++
bool* ptr;
*ptr = false;
if (ptr) {
    // will always be true without *
}
```

```c++
bool* ptr = &true; // error
```

```c++
int a = 5;
int* b = &a;
int c = *b // c is a separate location with value 5
```

```c++
int arr[10];
int* p6 = &arr[6];
int* p0 = &arr[0];

cout << (int)p6 << " " << (int)p0; // 162328 162304 (24 difference, 6 times 4(int))
cout <<"diff: " << p6 - p0; // 6 (pointers work on a space defined by the data type they point to)
```

```c++
int arr[10] = {3,6,9,12,15,18,21,24,27,30};
int* p0 = arr;

for (int i = 0; i < 10; i++) {
    cout << (arr + i) << endl; // 10 addresses that differ by 4 (int)
    cout << *(arr + i) << endl; // all array nums
    cout << p0 << " = " << *p0 << endl; // 10 addresses = each num
    p0++;
}
```

```c++
char word[] = "hello!";
char* p = word;
char* p0 = &word[0];
char* p3 = &word[3];

cout << p << endl; // hello! // char pointers are especially treated as strings, thats why 
cout << p0 << endl; // hello! // these prints dont show addresses
cout << p3 << endl; // lo!
```

```c++
// dynamic memory: when you know the size of memory only at run time, not at compile time
int size;
int* ptr;

cout << "Enter size: ";
cin >> size;

ptr = (int*)malloc(sizeof(int) * size);

for (int i = 0; i < size; i++) {
    cout << "Value: ";
    cin >> ptr[i];
}

for (int i = 0; i < size; i++) {
    cout << ptr[i] << " ";
}
```

```c++
int* ptr1;
int* ptr2;

ptr1 = (int*)malloc(10 * sizeof(int));
*ptr1 = { 0 }; // only sets ptr[0] = 0
for (int i = 0; i < 10; i++) {
    cout << ptr1[i] << " ";
}
// 0 -842150451 -842150451 -842150451 -842150451 -842150451 -842150451 -842150451 -842150451 -842150451
for (int i = 0; i < 10; i++) {
    ptr1[i] = i * 10;
}
cout << *(ptr1 + 5) << endl; // 50
```