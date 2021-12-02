- C/C++ function call syntax

```cpp
void func_name( int param, … );
for (int i = 0; i < SIZE; ++i) {
    func_name( param, … );
}
```

<br>

- CUDA kernel launch syntax
```cpp
__global__ void kernel_name( int param, … );
kernel_name <<< 1, SIZE >>>( param, … );
``` 
___<<<>>> 는 쿠다 컴파일러가 책임진다(C, Cpp 문법에 존재하지 않는 연산자)___

<br>

---

- cudaPeekAtLastError()
    - error flag를 가져오고, 리셋 X
    - 에러 확인용

<br>

- cudaGetLastError()
    - error flag를 가져오고, 리셋 O
    - 에러 처리용

<br>

---

- CUDA error check macro
```cpp
#define CUDA_CHECK_ERROR()	do { \
        cudaError_t e = cudaGetLastError(); \
        if (cudaSuccess != e) { \
            printf("cuda failure \"%s\" at %s:%d\n", \
                   cudaGetErrorString(e), \
                   __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)
```
___notice: do ~ while 뒤에 세미콜론 없음___<br>
___why? 매크로 함수 뒤에 세미콜론을 찍어도 문제없게(또는 꼭 찍어야 하도록) 만들고 싶다___<br>
___how? do ~ while(False)로 1회 실행되도록 묶어버리고 세미콜론 안쓰기 -> 코드에서 써야함___

<br>

---

- CUDA 에서의 kernel launch
    - many threads(ex. 1,000,000) on many core(ex. 1,000)가 일반적인 상황
     - 쓰레드 관리를 위한 모델 - 계층구조(launches are hierarchical, grid - block - thread)
        - 커널이 grid를 만들어서 grid가 실행되는 구조
        - grid는 많은 block들을, block들은 많은 thread들을 가짐.
            == thread가 묶여서 block, block이 묶여서 grid가 된다
    - thread 내부는 sequential execution.
        - 프로그래머들이 sequential programming에 워낙 익숙하기 때문
        - but, 모든 thread는 병렬로 실행되므로 병렬처리의 이점을 누릴 수 있음
    - grid, block 구조는 최대 3차원
        - kernel_func<<<dimGrid, dimBlock>>>(...);
        - kernelFunc<<<3, 4>>>(...);
        - kernelFunc<<<dim(3), dim(4)>>>(...);
        - kernelFunc<<<dim(3, 1, 1), dim(4, 1, 1)>>>(...);

<br>

---

- Vector Add
    - 1차원 배열 a와 b를 더해 를 구한다.
    - 검산 방법
        ```cpp
        float sumA = getSum(vecA, SIZE);
        float sumB = getSum(vecB, SIZE);
        float sumC = getSum(vecC, SIZE);
        float diff = fabsf(sumC - (sumA + sumB));
        //이론상 diff는 0.0이어야 하지만, floating point 연산 특성상 약간의 오차 발생
        //diff/SIZE 로 개당 오차를 구한 후 0에 접근하면 전체적인 계산이 옳다고 볼 수 있다.
        //getSum은 common.cpp확인. partial sum을 구한 후 다시 합치는 형태(float 타입 정밀도 문제)
        ```
    - 1M개 thread 동시 실행 불가능 에러 상황
        ```cpp
        __global__ void kernelVecAdd(float *c, const float *a, const float *b, unsigned n){
            unsigned i = threadIdx.x; // CUDA-provided index
            if (i<n){ 
                //boundary check!
                //혹시나 threadidx가 배열의 범위를 벗어나지 않았는지 체크
                //체크 해봤자 1clock!
                c[i]=a[i]+b[i];
            }
        }
        ...
        kernelVecAdd<<<1, SIZE>>>(dev_vecC, dev_vecA, dev_vecB, SIZE);
        cudaDeviceSynchronize();
        //invalid configuration argument.
        //SM이 1M개의 thread를 동시 실행 불가능. block당 1024개가 한계.
        ```
    - 위 상황 해결
        ```cpp
        __global__ void kernelVecAdd(float *c, const float *a, const float *b, unsigned n){
            unsigned i = blockIdx.x * blockDim.x + threadIdx.x; // CUDA-provided index
            if (i<n){ 
                c[i]=a[i]+b[i];
            }
        }
        ...
        kernelVecAdd<<<SIZE/1024, 1024>>>(dev_vecC, dev_vecA, dev_vecB, SIZE);
        cudaDeviceSynchronize();
        ```
    - cpp 구현시 kernel function
        - cpp template 적용 가능!
        - class member로 적용 불가 -> external void로 구현 후 멤버 함수가 호출
    - block은 32의 배수로 실행되지만, blockDim과 vector size가 맞아 떨어진다는 보장은 없음.
        - boundary check 구문과 함께 더 많은 thread를 구동한다.
        - `(dimBlock.x*dimGrid.x >= SIZE)` 를 보장해야 함
        - `올림(SIZE/dimBlock.x)` == `(A+(B-1))/B`
        ```cpp
        dim3 dimBlock( 1024, 1, 1 );
        dim3 dimGrid( (SIZE + dimBlock.x - 1) / dimBlock.x, 1, 1 );
        ```

<br>

---

- CUDA의 clock() kernel function
    - GPU에서의 clock ticks가 얼마나 되었는가? clock 횟수를 알려준다.
        ```cpp
        __global__ void kernelVecAcc(float *c, const float *a, const float *b, unsigned n, long long *times){
            clock_t start = clock();
            ...
            clock_t end = clock();
            if(i==0){ //굳이 전부 다 알 필요 없으면 0번 thread의 clock만 획득
                times[0] = (long long)(end - start);
            }
        }
        ```
    - clock 몇 번 튀었는가로 실제 시간을 알 수 있는건 아니므로, clock frequency를 알아야 한다.
        ```cpp
        int clk_freq = 1;
        cudaDeviceGetAttribute(&clk_freq, cudaDevAttrClockRate, 0);
        float elapsed_usec = clk_ticks * 1000.0F / clk_freq;
        ```

<br>

---

- AXPY( Z <- aX + Y )
    - X, Y, Z는 vector, a는 scalar 값
    - 그저 vector add 문제에 a를 곱하는 것
        - SAXPY : single precision(float)
        - DAXPY : double precision(double)
        - CAXPY : complex numbers(복소수)
    - 목표: multiply-add
        - MAC / MAD : multiply-accumulate / multiply-add
            - Assembly level에서 `round(round(a*x)+y)`를 한번에 하도록 구현
        - FMAC / FMA : fused multiply-accumulate / fused multiply-add
            - `round(a*x+y)`로 개선
            - 약간 빠름
            - 정밀도가 약간 올라감 -> deep learning 등에서 선호
            - __in one step ( machine instruction으로 구현 )__
                ```cpp
                float fmaf(float a, float x, float y);
                double fma(double a, double x, double y);
                //returns (a*x+y) with FAM instruciton
                ```
            - 똑똑한 컴파일러는 알아서 바꿔주기도 함..

<br>

- FMA의 의미?
    - 일부 연산 전형적인 FMA instruction으로 구현 가능.
    - 벡터의 내적 : dot product
        - a<sub>x</sub>*b<sub>x</sub> + a<sub>y</sub>*b<sub>y</sub> + a<sub>z</sub>*b<sub>z</sub>
        - ans <- 0
        - ans <- fma(a<sub>x</sub>, b<sub>x</sub>, ans)
        - ans <- fma(a<sub>y</sub>, b<sub>y</sub>, ans)
        - ans <- fma(a<sub>z</sub>, b<sub>z</sub>, ans)
        - accumulate(누적) 3회
    - 선형 보간법 : linear interpolation(lerp)
        - f(t) = (1-t)*v<sub>0</sub> + t*v<sub>1</sub> = (v<sub>0</sub> - t*v<sub>0</sub>) + t*v<sub>1</sub> [v<sub>0</sub> = (x<sub>0</sub>, y<sub>0</sub>) , v<sub>1</sub> = (x<sub>1</sub>, y<sub>1</sub>)]
        - fma(t, v<sub>1</sub>, fma(-t, v<sub>0</sub>, v<sub>0</sub>))
    - 사실 연산이 lerp 정도로 간단할 때는 큰 이득이 있지는 않다.. 더 복잡해질수록 차이가 남.

<br>

---

- CUDA hardware의 구조(Tesla GP100 예시)
    - 1GPU에 6GPC(graphics processing cluster)
    - 1GPC에 10Pascal SM -> 1GPU에 60SM
    - 1SM(unit) = 32SP + 16DP + 8SFU + 2Tex
        - SP(streaming processor) : FP32 core, 메인 CUDA core, ALU for a single CUDA thread
        - DP(double precision) : FP64 core
        - SFU(sepcial function unit) : sin, cos, square root 등 특별한 연산 1클락에 해결 가능
        - Tex(texture processor) : for graphics purpose, CUDA로 사용시 사용하지 않기도 하고 메모리로 쓰기도 함

<br>

- CUDA 의 확장성
    - CUDA dedvice는 1~4개의 SM의 저가 모바일 기기부터 1000+의 고가 워크스테이션까지 매우 다양
    - thread block 개념을 도입하여 해결(SM 1개가 thread block 1개 처리)
    - so, grid - block - thread의 계층 구조 필요
    - thread block 들이 SM에 자유롭게 assign 되어서 처리되는 구조
    - Each block can execute in any order relative to other blocks

<br>

- SM에서 CU(control Unit, SM당 1개)의 실행 구조
    - 1개의 CU의 제어를 받아 32 core(SP) 가 물리적으로 동시에 실행
    - 1개의 warp scheduler
    - 32 thread가 같은 instruction을 동시 실행
    - SM 1개는 2048+ thread를 동시 관리 -> memory의 느린 반응 속도 해결

<br>

- Thread와 Warp
    - Thread는 독립적 실행 단위(실)
    - Warp 평행하게 관리되는 여러개의 실(Warp를 만드는 것처럼 여러 실을 평행하게 관리)
    - CUDA에서의 Warp는 32개의 thread(SM이 32개의 SP를 가지므로)
    - lane: Warp 내에서의 thread의 index(0~31)
    - block 에는 1024개의 thread가 있지만, 32개씩 끊어서 warp로 관리
    - 20개 이상의 warp가 대기 상태로 있는 것이 효율적
        - memory access 시간을 고려
        - warp 전환간 거의 zero-overhead. 충분히 많은 register를 확보하고 있기 때문
        - warp scheduler는 HW로 구현되어 오버헤드 거의 없음

<br>

- 2레벨 병렬 처리
    - grid는 thread blocks로 이루어져 있으므로 SM에 병렬 처리
    - thread block은 여러 warp로 갈라져서 병렬 처리
    - warp / block 종료 시 다음 warp / block을 처리
    - 자원 제약에 대한 고려가 필요하지만, thread수를 1024정도로 잡으면 문제없음
    - block의 실행 순서가 정해져 있지 않음

<br>

- warp id, lane id
    - GPU assembly instruction으로 체크 가능
    - warp id : SM 내에서, 특정 warp의 ID number
        ```cpp
        __device__ unsigned warp_id(void) {
            // this is not equal to threadIdx.x / 32
            unsigned ret;
            asm volatile ("mov.u32 %0, %warpid;" : "=r"(ret));
            return ret;
        }
        ```
    - lane id : warp 내에서, 자신의 lane id
        ```cpp
        __device__ unsigned lane_id(void) {
            unsigned ret;
            asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
            return ret;
        }
        ```

<br>

---

- Matrix의 표현
    - nrow : number of rows(행)
    - ncol : number of cols(열)
    - (row, col) or (y, x)

<br>

- Matrix Addition Concept
    - c<sub>ij</sub> = a<sub>ij</sub> + b<sub>ij</sub>
    - 본질은 Vector Addition과 크게 다르지 않다.
    - 다만, 2차원 matrix를 physical 관점에서 1차원에 어떻게 잘 저장할 것인가?
    - 1차원 메모리를 2차원으로 해석하여 연산한다.

<br>

- Matrix Layout (in C/C++)
    - logical
        <table>
            <tr>
                <td>M<sub>0,0</sub></td>
                <td>M<sub>0,1</sub></td>
                <td>M<sub>0,2</sub></td>
                <td>M<sub>0,3</sub></td>
            </tr>
            <tr>
                <td>M<sub>1,0</sub></td>
                <td>M<sub>1,1</sub></td>
                <td>M<sub>1,2</sub></td>
                <td>M<sub>1,3</sub></td>
            </tr>
            <tr>
                <td>M<sub>2,0</sub></td>
                <td>M<sub>2,1</sub></td>
                <td>M<sub>2,2</sub></td>
                <td>M<sub>2,3</sub></td>
            </tr>
            <tr>
                <td>M<sub>3,0</sub></td>
                <td>M<sub>3,1</sub></td>
                <td>M<sub>3,2</sub></td>
                <td>M<sub>3,3</sub></td>
            </tr>
        </table>
    - physical
        - `M = &(M[0][0])`
        - `M[y][x]`
        <table>
            <tr>
                <td>M<sub>0,0</sub></td>
                <td>M<sub>0,1</sub></td>
                <td>M<sub>0,2</sub></td>
                <td>M<sub>0,3</sub></td>
                <td>M<sub>1,0</sub></td>
                <td>M<sub>1,1</sub></td>
                <td>M<sub>1,2</sub></td>
                <td>M<sub>1,3</sub></td>
                <td>M<sub>2,0</sub></td>
                <td>M<sub>2,1</sub></td>
                <td>M<sub>2,2</sub></td>
                <td>M<sub>2,3</sub></td>
                <td>M<sub>3,0</sub></td>
                <td>M<sub>3,1</sub></td>
                <td>M<sub>3,2</sub></td>
                <td>M<sub>3,3</sub></td>
            </tr>
        </table>
    - physical(re-interpret)
        - `M = cudaMalloc(...)`
        - `M[y*WIDTH + x]` -> `M[i]`
        <table>
            <tr>
                <td>M<sub>0,0</sub></td>
                <td>M<sub>0,1</sub></td>
                <td>M<sub>0,2</sub></td>
                <td>M<sub>0,3</sub></td>
                <td>M<sub>1,0</sub></td>
                <td>M<sub>1,1</sub></td>
                <td>M<sub>1,2</sub></td>
                <td>M<sub>1,3</sub></td>
                <td>M<sub>2,0</sub></td>
                <td>M<sub>2,1</sub></td>
                <td>M<sub>2,2</sub></td>
                <td>M<sub>2,3</sub></td>
                <td>M<sub>3,0</sub></td>
                <td>M<sub>3,1</sub></td>
                <td>M<sub>3,2</sub></td>
                <td>M<sub>3,3</sub></td>
            </tr>
            <tr>
                <td>M<sub>0</sub></td>
                <td>M<sub>1</sub></td>
                <td>M<sub>2</sub></td>
                <td>M<sub>3</sub></td>
                <td>M<sub>4</sub></td>
                <td>M<sub>5</sub></td>
                <td>M<sub>6</sub></td>
                <td>M<sub>7</sub></td>
                <td>M<sub>8</sub></td>
                <td>M<sub>9</sub></td>
                <td>M<sub>10</sub></td>
                <td>M<sub>11</sub></td>
                <td>M<sub>12</sub></td>
                <td>M<sub>13</sub></td>
                <td>M<sub>14</sub></td>
                <td>M<sub>15</sub></td>
            </tr>
        </table>

<br>

- Matrix Addition 에서 Thread Block 설계 (10,000 x 10,000)
    - 1개 Thread Block 에 1024개의 thread를 선호 -> `32 x 32 thread block`
    - 10,000 / 32 = 312.5 -> `313 x 313 grid size`
    - 32 * 313 = 10,016 -> `10,016 x 10,016 total threads`
        ```cpp
        dim3 dimBlock(32, 32, 1);
        dim3 dimGrid((ncol + dimBlock.x - 1) / dimBlock.x, (nrow + dimBlock.y - 1) / dimBlock.y, 1);
        ```
    - 범위를 넘어가는 16개 thread들은 메모리 접근 없이 discard !
        ```cpp
        __global__ void kernel_matadd( float* c, const float* a, const float* b, unsigned nrow, unsigned ncol ) {
            unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
            unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
            if (row < nrow && col < ncol) { // 넘어가면 discard !
                unsigned i = row * ncol + col; // converted to 1D index
                c[i] = a[i] + b[i];
            }
        }
        ...
        cudaMalloc( (void**)&dev_matA, nrow * ncol * sizeof(float) );
        ...
        kernel_matadd <<< dimGrid, dimBlock>>>( dev_matC, dev_matA, dev_matB, nrow, ncol );

        ```
    - x, y가 스왑되어 버리면 논리적으로는 문제가 없지만, 접근할 메모리 공간이 불연속적이 되어(흩어짐) 속도가 확 떨어진다.

<br>

---

- 왜 연속된 메모리 공간에 접근하면 DRAM의 속도가 빨라지는가?
    - DRAM은 capacitor에 bit를 저장하며, 평면상에 2D array로 배열
    - bit 하나를 가져오는건 2차원 형태에서 특정 위치를 찾는 문제
    - DRAM 내부적으로는 한 줄 전체를 가져온 다음 하나만 골라서 준 것.
    - 여러(8/16/32) bit를 한번에 가져와야 하므로, 여러 칩을 묶어서 사용
    - 다음 번지수에 접근한다면(sequential), 이미 가져온 것을 그대로 준다(DRAM Bursting)
    - DDR(double data rate): CPU가 어떤 주소를 요구할 때, 다음 주소를 요구할 것이라 가정하고 바로 준다.
    - Dual/Triple/Quad Channel: channel 별로 주소를 교대로 할당하여 접근 속도를 높임
    - __Memory Coalescing(합병)__

<br>

- Memory Coalescing
    - memory 효율을 높이는 방법
    - chunk 단위로 메모리를 주고받음
    - CUDA는 128byte의 chunk를 주고받음. 
    - `128byte = 32thread(1warp) * 4byte(float, int)` R/W per warp !
    - cudaMAlloc(): 언제나 256byte 단위의 boundary를 갖도록 할당
    - C++11의 경우, 코드에 alignment boundary 명시 가능
        ```cpp
        int main(){
            alignas(16) int a[4];   // 16byte boundary aligned
            alignas(1024) int b[4]; // 1024byte boundary aligned
            printf("%p\n", a);
            printf("%p\n", b);
            assert(alignof(a) == 16);   // always true
            assert(alignof(b) == 1024); // always true
            if(alignof(b) != 1024) exit(0); // fatal !
        }

        struct alinas(16) sse_t { // structures, classes 와 사용 가능
            float sse_data[4];
        };

        struct alignas(8) S { ... };
        struct alignas(1) U { S s; }; // error: U should be aligned to 8 or larger
        ```

<br>

- Pitch의 효용
    - transaction size에 따른 Memory Access 필요 횟수
        |transaction size|fully aligned|misaligned|delay|
        |-|-|-|-|
        |<center>32B|<center>4회|<center>5회|<center>20%|
        |<center>64B|<center>2회|<center>3회|<center>50%|
        |<center>128B|<center>1회|<center>2회|<center>100%|
    - Thread Block처럼, 뒤의 메모리 공간을 낭비(discard)하기 위해 pitch를 두어 fully aligned 하게 만들어 준다.
    - `cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height)`
        - allocate pitched memory on device
            |parm|description|
            |-|-|
            |devPTr|pointer to allocated pitched device memory|
            |pitch|pitch for allocation (__in bytes__)|
            |width|requested pitched allocation width (__in bytes__)|
            |height|requested pitched allocation height|
        - pitch와 width는 data type가 무엇이 될 지 모르므로, 단위는 byte
        - 2D Matrix에서 (row, col) 원소의 주소 계산
            - `T* pElem = (T*)((char*)baseAddr + row * pitch) + col;`
            - width 대신 pitch를 사용
            - 1byte를 쓰는 char*로 캐스팅 이용
    - `cudaMemcpy2D ( void* dst, size_t dpitch,
const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind )`
        - pitched allocation 간의 memory copy
            |parm|description|
            |-|-|
            |dst, dpitch|destination, with pitch (__in byte__)|
            |src, spitch|source, with pitch (__in byte__)|
            |width, height|matrix size (__in byte__)|
            |kind|type of transfer (as in cudaMemcpy|

<br>

- Pitch를 고려한 Matrix Addition
    - array index를 계산할 때, byte 단위로 pitch를 고려해야 한다. (offset 이용)
        ```cpp
        __global__ void kernel_matadd( float* c, const float* a, const float* b,
                                unsigned nrow, unsigned ncol, size_t dev_pitch ) {
            register unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
            if (col < ncol) {
                register unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
                if (row < nrow) {
                    register unsigned offset = row * dev_pitch + col * sizeof(float); // in byte
                    *((float*)((char*)c + offset)) = *((const float*)((const char*)a + offset))
                                                    + *((const float*)((const char*)b + offset));
                }
            }
        }
        ```
    - malloc 및 memcpy
        ```cpp
        size_t dev_pitch = 0;
        cudaMallocPitch( (void**)&dev_matA, &dev_pitch, ncol * sizeof(float), nrow );
        ...
        cudaMemcpy2D( dev_matA, dev_pitch, matA, host_pitch, ncol * sizeof(float), nrow, cudaMemcpyHostToDevice);
        ...
        kernel_matadd <<< dimGrid, dimBlock>>>( dev_matC, dev_matA, dev_matB, nrow, ncol, dev_pitch );
        cudaDeviceSynchronize();
        ...
        cudaMemcpy2D( matC, host_pitch, dev_matC, dev_pitch, ncol * sizeof(float), nrow, cudaMemcpyDeviceToHost);
        ...
        ```

<br>

---

- 3D Array Layout (in C/C++)
    - 3D
        - z<sub>0</sub>
            <table>
                <tr>
                    <td>M<sub>0,0,0</sub></td>
                    <td>M<sub>0,0,1</sub></td>
                    <td>M<sub>0,0,2</sub></td>
                </tr>
                <tr>
                    <td>M<sub>0,1,0</sub></td>
                    <td>M<sub>0,1,1</sub></td>
                    <td>M<sub>0,1,2</sub></td>
                </tr>
                <tr>
                    <td>M<sub>0,2,0</sub></td>
                    <td>M<sub>0,2,1</sub></td>
                    <td>M<sub>0,2,2</sub></td>
                </tr>
            </table>
        - z<sub>1</sub>
            <table>
                <tr>
                    <td>M<sub>1,0,0</sub></td>
                    <td>M<sub>1,0,1</sub></td>
                    <td>M<sub>1,0,2</sub></td>
                </tr>
                <tr>
                    <td>M<sub>1,1,0</sub></td>
                    <td>M<sub>1,1,1</sub></td>
                    <td>M<sub>1,1,2</sub></td>
                </tr>
                <tr>
                    <td>M<sub>1,2,0</sub></td>
                    <td>M<sub>1,2,1</sub></td>
                    <td>M<sub>1,2,2</sub></td>
                </tr>
            </table>
        - z<sub>2</sub>
            <table>
                <tr>
                    <td>M<sub>2,0,0</sub></td>
                    <td>M<sub>2,0,1</sub></td>
                    <td>M<sub>2,0,2</sub></td>
                </tr>
                <tr>
                    <td>M<sub>2,1,0</sub></td>
                    <td>M<sub>2,1,1</sub></td>
                    <td>M<sub>2,1,2</sub></td>
                </tr>
                <tr>
                    <td>M<sub>2,2,0</sub></td>
                    <td>M<sub>2,2,1</sub></td>
                    <td>M<sub>2,2,2</sub></td>
                </tr>
            </table>
    - 1D
        - M[z][y][x] : 3D array size = depth x height x width
        - i = (z * HEIGHT + y) * WIDTH + x;
        <table>
            <tr>
                <td>M<sub>0,0,0</sub></td>
                <td>M<sub>0,0,1</sub></td>
                <td>M<sub>0,0,2</sub></td>
                <td>M<sub>0,1,0</sub></td>
                <td>M<sub>0,1,1</sub></td>
                <td>M<sub>0,1,2</sub></td>
                <td>M<sub>0,2,0</sub></td>
                <td>M<sub>0,2,1</sub></td>
                <td>M<sub>0,2,2</sub></td>
                <td>M<sub>1,0,0</sub></td>
                <td>M<sub>1,0,1</sub></td>
                <td>M<sub>1,0,2</sub></td>
                <td>M<sub>1,1,0</sub></td>
                <td>M<sub>1,1,1</sub></td>
                <td>M<sub>1,1,2</sub></td>
                <td>M<sub>...</sub></td>
            </tr>
            <tr>
                <td>M<sub>0</sub></td>
                <td>M<sub>1</sub></td>
                <td>M<sub>2</sub></td>
                <td>M<sub>3</sub></td>
                <td>M<sub>4</sub></td>
                <td>M<sub>5</sub></td>
                <td>M<sub>6</sub></td>
                <td>M<sub>7</sub></td>
                <td>M<sub>8</sub></td>
                <td>M<sub>9</sub></td>
                <td>M<sub>10</sub></td>
                <td>M<sub>11</sub></td>
                <td>M<sub>12</sub></td>
                <td>M<sub>13</sub></td>
                <td>M<sub>14</sub></td>
                <td>M<sub>...</sub></td>
            </tr>
        </table>
    - 코드 선언
        ```cpp
        dim3 dimImage( 300, 300, 256 ); // x, y, z order - width (ncolumn), height (nrow), depth
        ...
        matA = new float[dimImage.z * dimImage.y * dimImage.x];
        matB = new float[dimImage.z * dimImage.y * dimImage.x];
        matC = new float[dimImage.z * dimImage.y * dimImage.x];
        ```

<br>

- filter 연산 kernel 함수 비교
    - host (3중 for문)
        ```cpp
        for (register unsigned z = 0; z < dimImage.z; ++z) {
            for (register unsigned y = 0; y < dimImage.y; ++y) {
                for (register unsigned x = 0; x < dimImage.x; ++x) {
                    unsigned i = (z * dimImage.y + y) * dimImage.x + x; // convert to 1D index
                    matC[i] = matA[i] * matB[i];
                }
            }
        }
        ```
    - device
        - 300 x 300 x 256 (처리해야 할 데이터)
        - 8 x 8 x 8 = 512 thread per block (전방체 형태)
        - 38 x 38 x 32 thread blocks (thread per block로 도출)
            ```cpp
            dim3 dimGrid(div_up(dimImage.x, dimBlock.x), div_up(dimImage.y, dimBlock.y), div_up(dimImage.z, dimBlock.z));
            //div_up 는 ceil(올림) (lhs+rhs-1)/rhs 연산
            ```
            
        ```cpp
        __global__ void kernel_filter( float* c, const float* a, const float* b, unsigned ndim_z, unsigned ndim_y, unsigned ndim_x ) {
            unsigned idx_z = blockIdx.z * blockDim.z + threadIdx.z;
            unsigned idx_y = blockIdx.y * blockDim.y + threadIdx.y;
            unsigned idx_x = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx_x < ndim_x && idx_y < ndim_y && idx_z < ndim_z) { // 정확하게 들어오는지 모두 확인.
                unsigned i = (idx_z * ndim_y + idx_y) * ndim_x + idx_x; // converted to 1D index
                c[i] = a[i] * b[i];
            }
        }
        ```
<br>

- filter 연산 with Pitched Matrix
    - kernel
        - pitch를 추가로 인자로 받는다.
        - offset을 byte단위로 처리한다.
        ```cpp
        __global__ void kernel_filter( void* matC, const void* matA, const void* matB, size_t pitch, unsigned ndim_z, unsigned ndim_y, unsigned ndim_x ) {
            register unsigned idx_z = blockIdx.z * blockDim.z + threadIdx.z;
            register unsigned idx_y = blockIdx.y * blockDim.y + threadIdx.y;
            register unsigned idx_x = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx_x < ndim_x && idx_y < ndim_y && idx_z < ndim_z) {
                register unsigned offset_in_byte = (idx_z * ndim_y + idx_y) * pitch + idx_x * sizeof(float);
                *((float*)((char*)matC + offset_in_byte))
                    = *((const float*)((const char*)matA + offset_in_byte))
                    * *((const float*)((const char*)matB + offset_in_byte));
            }
        }
        ```
    - 3D Pitched Matrix 함수
        - `cudaMalloc3D( cudaPitchedPtr* pitchedDevPtr, cudaExtent extent );`
            - allocates logical 1D/2D/3D memory objects on the device
                |parm|description|
                |-|-|
                |pitchedDevPtr|pointer to the allocated device memory|
                |extent|requested allocation size|
        - `cudaExtent make_cudaExtent( size_t w, size_t h, size_t d );`
            - make a CUDA extent ↑
                |parm|description|
                |-|-|
                |w|width (__in byte__)|
                |h|height (__in elements__)|
                |d|depth (__in elements__)|
        - `cudaPitchedPtr`
            - 일반적으로 cudaMalloc3D가 만들어 주는 것이 일반적
            ```cpp
            struct cudaPitchedPtr {
                size_t pitch;   // pitch in bytes
                void* ptr;      // real pointer
                size_t xsize;   // width in bytes
                size_t ysize;   // height in elements
            };
            ```
        - `cudaPitchedPtr make_cudaPitchedPtr( void* d, size_t p, size_t xsz, size_t ysz );`
            - 이미 만들어진 ptr을 pitched ptr로 만들어 준다.
            - C/Cpp에서 new로 만든 ptr을 CUDA에서도 사용해야 할 때 등
                |parm|description|
                |-|-|
                |d|pointer to allocated memory|
                |p|pitch of allocated memory (__in bytes__)|
                |xsz|width of allocation (__in bytes__)|
                |ysz|height of allocation (__in elements__)|
        - `make_cudaPos( size_t x, size_t y, size_t z );`
            - 3D array 내에서 특정 위치를 지정
        - `cudaMemcpy3D( const cudaMemcpy3DParams* p );`
            - 파라미터가 복잡하여 별도의 struct를 만들어 이용함.
            - struct 구조
            ```cpp
            struct cudaExtent { size_t depth, height, width; }; // copy 크기
            struct cudaPos { size_t x, y, z; }; // copy 위치
            struct cudaMemcpy3DParms {
                struct cudaArray* srcArray;
                struct cudaPos srcPos;
                struct cudaPitchedPtr srcPtr;
                struct cudaArray* dstArray;
                struct cudaPos dstPos;
                struct cudaPitchedPtr dstPtr;
                struct cudaExtent extent;
                enum cudaMemcpyKind kind;
            };
            ```
    - cudaMemcpy3D 예시
        ```cpp
        ...
        matA = new float[dimImage.z * dimImage.y * dimImage.x];
        struct cudaPitchedPtr pitchedA = make_cudaPitchedPtr( matA, dimImage.x * sizeof(float), dimImage.x * sizeof(float), dimImage.y );
        ...
        struct cudaExtent extentInByte = make_cudaExtent( dimImage.x * sizeof(float), dimImage.y, dimImage.z );
        struct cudaPitchedPtr dev_pitchedA = { 0 };
        cudaMalloc3D( &dev_pitchedA, extentInByte );
        ...
        struct cudaPos pos_origin = make_cudaPos( 0, 0, 0 );
        struct cudaMemcpy3DParms paramA = { 0 };
        paramA.srcPos = pos_origin;
        paramA.srcPtr = pitchedA;
        paramA.dstPos = pos_origin;
        paramA.dstPtr = dev_pitchedA;
        paramA.extent = extentInByte;
        paramA.kind = cudaMemcpyHostToDevice;
        cudaMemcpy3D( &paramA );
        ...
        ```
    - 간단한 연산에서는 큰 이득을 못 볼 수도 있다.

<br>

---

- CUDA의 메모리 계층 구조
    |name|description|r/w 대상|
    |-|-|-|
    |registers|SP의 레지스터. SP의 작업공간|thread|
    |shared memory|SM의 캐시. 같은 block의 쓰레드들이 공유|block|
    |global memory|메인 메모리.|grid|
    |local memory|global memory를 thread가 일부 독립적으로 이용|thread|
    |constant memory|읽기 전용으로 속도를 높임|grid|
    - SM간에는 직접적인 통신 사실상 불가능. global memory 이용
    - 현대적인 CUDA 에서는 local memory를 shared memory에 잡는 경우도 있음
    - 몇 개의 레지스터를 이용할 수 있는지, shared memory 를 얼마나 쓸 수 있는지 확인하여 더 최적화된 프로그램 작성 가능
        - 레지스터 수를 넘어가는 변수 정의 시 local memory에 잡혀 속도 저하 이슈 등등
        - Windows: NVIDIA GPU Computing Toolkit\CUDA\vX.Y\extras\demo_suite\deviceQuery.exe
        - Linux: cuda-11.3\extras\demo_suite\deviceQuery

<br>

- Shared Memory in CUDA
    - 레지스터만큼 빠름(cache memory)
    - SM의 공통의 영역. 같은 block 내의 thread끼리는 공유 가능
    - 캐시를 직접 제어 가능하므로, 효과적으로 사용하면 속도를 높일 수 있다
    - 레지스터에는 배열 선언이 불가능하지만, shared memory에는 선언 가능

<br>

- CUDA variable type qualifiers
    |variable declaration|memory|scope|life time|penalty|
    |-|-|-|-|-|
    |`int var;`|register|thread|thread|1x|
    |`int array[10];`|local|thread|thread|100+x|
    |`__device__ __shared__ in shared_var;`|shared|block|block|1x|
    |`__device__ int global_var;`<br>`cudaMalloc(&dev_ptr, size)`|global|grid|application|100+x|
    |`__device__ __constant__ int constatn_var;`|constant|grid|application|1x|
    - `__shared__`, `__constant__`와 함께 쓸 때 `__device__`는 붙여도 되고 안붙여도 된다.

    - 선언 위치
        ```cpp
        __device__ float global_array[1024]; //global memory
        __constant__ float coefficient_a = 1.23F; //constant memory
        __global__ void kernelFunc(float* dst, const float* src) {
            float p = src[threadIdx.x]; //register
            float heap[10]; //local memory
            __shared__ float partial_sum[1024];//shared memory
            ...
        }
        ```

- device에 직접 배열 선언해서 활용하기
    ```cpp
    ...
    __device__ float dev_x[vecSize]; // device에 array 선언
    __device__ float dev_y[vecSize];
    __device__ float dev_z[vecSize];
    ...
    __global__ void kernelSAXPY( unsigned n ) { // memory address 계산할 필요 없어짐.
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            dev_z[i] = dev_a * dev_x[i] + dev_y[i];
        }
    }
    ...
    cudaMemcpyToSymbol( dev_x, host_x, sizeof(host_x) );
    ...
    cudaMemcpyFromSymbol( host_z, dev_z, sizeof(host_z) );
    ```
    - `cudaMemcpyToSymbol( dev_x, host_x, sizeof(host_x) );`
        - `__device__`로 선언된 array 이름을 받아 device로 복사한다.
    - `cudaMemcpyFromSymbol( host_z, dev_z, sizeof(host_z) );`
        - `__device__`로 선언된 array 이름을 받아 host로 복사한다.
    - 이렇게 하면 편한데 왜 cudaMalloc을 썼나?
        - 너무 큰 배열을 선언하면 static 변수 한도를 넘어버려 compiler 이슈 발생 -> 동적 할당을 쓰자
    - `cudaGetSymbolAddress (void** devPtr, const void* symbol );`
        - 미리 선언한 cuda array의 주소 획득
            ```cpp
            void* ptr_x = nullptr;
            cudaGetSymbolAddress( &ptr_x, dev_x );
            cudaMemcpy( ptr_x, host_x, vecSize * sizeof(float), cudaMemcpyHostToDevice );
            ```

<br>

---

- Shared memory 사용 전략
    - 절차
        1. 데이터는 일단 global 메모리에 적재
        2. shared memory로 copy
        3. thread block 내에서 여러 스레드가 협업하여 가공
        4. 계산 결과를 global memory로 copy
    - kind of divide and conquer approach
    - race condition !
        - 내가 써주고, 남이 써준 데이터를 읽을 때 주의
        - 각 warp가 진행한 정도가 다를 수 있음

- Barrier Synchronization
    - CUDA intrinstic functions <sub>내장 함수</sub>
        - instrinsic <sub>내재하는, 본질적인</sub> function : 컴파일러가 해당 위치아 특별한 code를 생성(macro 쓰는 것 처럼)
        - function call이 일어나지는 않음
    - `__syncthreads()`
        - 모든 쓰레드가 이 위치까지 와 있기를 바람
        - 울타리(경계)에서 쓰레드들을 모두 세워버림
        - time sharing 관점에서, 쓰레드를 CPU에 넣지 않으면 실행되지 않음(기다림)
        ```cpp
        __global__ void kernel_func(void){
            __shared__ float shared[SIZE];
            ...
            shared[i] = value;  // write
            __syncthreads();    //모든 데이터가 target 영역에 준비되었음을 확신 가능
            ...
            another = shared[j]; // read
            ...
        }
        ```
        - heavy operation. 필요할 때만 쓸 것

<br>

- Adjacent Diffrence
    - 인접한 두 원소들 사이의 차이 계산
    - `vecB[i] = vecA[i] - vecA[i-1]` (첫 원소의 경우는 예외로 0)
    - host(single for loop)
        ```cpp
        for (register unsigned i = 0; i < num; ++i) {
            if (i == 0)
                vecB[i] = vecA[i] - 0.0f; // special case for i = 0
            else
                vecB[i] = vecA[i] - vecA[i - 1]; // normal case
        }
        ```
    - device
        ```cpp
        __global__ void kernelAdjDiff( float* b, const float* a, unsigned num ) {
            unsigned i = blockIdx.x * blockDim.x + threadIdx.x; // CUDA-provided index
            if (i == 0)
                b[i] = a[i] - 0.0f;
            else if (i < num)
                b[i] = a[i] - a[i - 1];
        }
        ```
    - shared version
        - `__syncthreads()` 사용
        - threadIdx가 0인 경우, shared memory X, global memory에서 읽기
        - Tiled Approach. 조각조각난 같은 크기의 타일을 붙여서 전체를 커버
            - shared memory로 들고 온 thread block의 데이터를 tile이라 할 수 있다(공식 용어는 아님)
            - global index와 local index을 따로 찾아서 계산해야 맞아 들어간다
        ```cpp
        __global__ void kernelAdjDiff(float* b, const float* a, unsigned num) {
            __shared__ float s_data[1024];
            register unsigned tx = threadIdx.x; //local index
            register unsigned i = blockIdx.x * blockDim.x + threadIdx.x; // global index
            if (i < num) {
                s_data[tx] = a[i];
                __syncthreads();
                if (tx > 0) 
                    b[i] = s_data[tx] - s_data[tx - 1];
                else if (i > 0) //local index == 0
                    b[i] = s_data[tx] - a[i - 1];
                else // global index == 0
                    b[i] = s_data[tx] - 0.0f;
            }
        }
        ```
    - `__syncthreads()` overuse
        - data update 이후 syncthreads 사용
        - 속도 저하를 가져옴. 하지 말 것 !
        ```cpp
        __global__ void kernelAdjDiff(float* b, const float* a, unsigned num) {
            __shared__ float s_data[1024];
            register unsigned tx = threadIdx.x;
            register unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < num) {
                s_data[tx] = a[i];
                __syncthreads();
                if (tx > 0) 
                    b[i] = s_data[tx] - s_data[tx - 1];
                else if (i > 0)
                    b[i] = s_data[tx] - a[i - 1];
                else
                    b[i] = s_data[tx] - 0.0f;
                __syncthreads(); // data update 이후의 syncthreads
            }
        }
        ```
        - 가능하면 적은 syncthreads

<br>

- Shared memory의 flexible size
    1. kernel 에서 extern으로 선언
        - 배열이 하나 있는데, size는 몰라요(array라는 사실만)
        - `extern __shared__ float s_data[];`
    2. kernel launch시에 __memory size in byte__ 선언
        - `kernelAdjDiff<<<dimGrid, dimBlock, sizeInByte>>>(...)`
        - thread block이 sizeInByte만큼의 shared memory 를 allocate
        - dynamic allocate으로, 메모리 공간 잡는 시간 소요(약간의 속도 저하)
    ```cpp
        __global__ void kernelAdjDiff(float* b, const float* a, unsigned num) {
            extern __shared__ float s_data[]; // sxtern 으로 선언
            register unsigned tx = threadIdx.x;
            register unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < num) {
                s_data[tx] = a[i];
                __syncthreads();
                if (tx > 0) 
                    b[i] = s_data[tx] - s_data[tx - 1];
                else if (i > 0)
                    b[i] = s_data[tx] - a[i - 1];
                else
                    b[i] = s_data[tx] - 0.0f;
            }
        }
        ...
        kernelAdjDiff<<<dimGrid, dimBlock, blocksize*sizeof(float)>>>(dev_vecB, dev_vecA, num);
        // blocksize에 Byte수를 계산하여 넘겨준다.
        ...
    ```

<br>

- Device Query
    - 가용 shared memory size 계산
        <table>
            <tr>
                <td>Total amount of shared memory per block</td>
                <td>49152 bytes</td>
            </tr>
            <tr>
                <td>Maximum number of threads per multiprocessor</td>
                <td>1024</td>
            </tr>
            <tr>
                <td>Maximum number of threds per block</td>
                <td>1024</td>
            </tr>
        </table>

        - 블럭당 48KB 제공, 블럭당 1024 쓰레드 -> 쓰레드당 48B 제공
        - 쓰레드당 48B 제공, float 당 4B -> 쓰레드당 float 12개 정도 사용 가능
        - 관련 정보들 api 함수로 조회(Device Query) 가능
    - `cudaGetDeviceProperties(cuda DeviceProp* prop, int device)`
        ```cpp
        ...
        cudaGetDeviceCount(&deviceCount);
        cudaGetDeviceProperties(&deviceProp, 0);
        blocksize = deviceProp.maxThreadsPerBlock;
        ...
        kernelAdjDiff<<<dimGrid, dimBlock, blocksize*sizeof(float)>>>(...);
        ...
        ```
        - Device Query가 오랜 시간이 걸리지는 않고, dynamic allocation시 좀 걸린다.

<br>

- 최종 속도 비교
    |method|time(usec)|
    |-|-|
    |host|31,810|
    |global memory|473|
    |shared memory|587|587|
    |dynamic allocation|594|
    |2 `syncthreads()`|629|

    - 의외로 global memory를 사용하는 쪽이 더 빠른 이유
        - 최신 CUDA device들은 L1/L2 cache의 추가 등으로 매우 성능이 좋음
        - 계산이 복잡한 경우에는 확실히 빨라짐.(특히 matrix 다루는 등)
        - device 마다 테스트 해 볼 필요 있음.(CUDA 버전이 올라가면서 또 바뀔수도)

<br>

---

- kernel function 내에서의 pointer
    - 포인터라는 사실은 알고 있지만, 어느 memory space를 가리키는지는 알 수 없음!
    - `__shared__ int* ptr;`의 의미
        - ptr variable이 shared memory에 위치
        - ptr이 가리키는 것이 shared memory라는 것은 절대 아님! (어디든 가리킬 수 있음)
    - 같은 warp 내에서 pointer가 다른 memory 영역을 access 하는 경우
        - CUDA system crash or 성능 저하
        - `Warning: Cannot tell what pointer points to, assuming global memory space`
    - 가능하면 simple(간단)하고 regular(예상가능)하게
    - pointer to pointer 피하기
        - 링크드 리스트, 트리 등 지양

<br>

- kernel function의 parameter의 memory space
    - 기본적으로 call by value(struct 넘길 시 그대로 copy)
    - pointer값은 모두 CUDA global memory space로 추정
        - 배열을 넘겨도 모두 global memory에 있다 가정하고 찾는다.
        - host space의 배열을 kernel function에 쓰면 바로 error!