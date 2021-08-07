# Notes

## 1. Pointers

- `&y` means the memory direction of y.


```c++
char c;
int* pc; // error
```

```c++
char c;
int* pc; // no error
```

- Indirection is when you take out the value of the pointer's target, `cout<<*py`.

```c++
float y = 2.5;
float *py = &y; // needs to be float because y is float
float **py = &py; // direction of the pointer py, needs to be float because of y
```

- Void Pointers can point to whatever type of value. Also called a generic pointer.

- A null pointer does not point to any direction, does not point to trash.

```c++
char c;
char* p = NULL;
```

## Constant Pointers (int *const)

```c++
int x = 4;
int* const px = &x; // pointer cannot change its memory address (but x can change), and must be initialized
cout << *px; // 4
*px = 8; // x or *px can change
cout << *px; // 8
// error px = &y;
```

## Pointers To Constants (const *int)

```c++
const int x_const = 12;
// error int* p3 = &x_const; 
const int* p3 = &x_const; // no error, also this pointer can point to different const int variables
*p3 = 11 // error bc you're changing x_const or p3
const int y_const = 10;
p3 = &y_const; // p3 can change its target, but its variable type is const int
```

## Pointers To Arrays

```c++
int arr[3] = {5,7,9};
int *p = arr; // same as p = &arr[0]
cout << *p; // 5
cout << p[0]; // 5
cout << *(p + 1); // 7, bc we are advancing one memory cell
```

## Exercise 1

```c++
#include <iostream>
using namespace std;

void func(int *p1, int *p2, int *pr) {
    
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
}
```

## 2. Dynamic Memory

- `malloc()` returns a void* pointer and so we cast it `(int *)`.

```c++
int *p;
p = (int*)malloc(sizeof(int));
// for arrays (size 10): p = (int*)malloc(sizeof(int) * 10);
*p = 45
cout<<*p;
free(p);
```

## Exercise 2

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