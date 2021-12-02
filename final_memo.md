- Matrix Copy
    - 행렬을 같은 크기의 행렬로 복사. Source -> Target
    - 행렬 복사는 행렬 연산에서 `속도에 있어 이론적 한계`가 된다.
        - 아무 연산도 안할 때(복사) 제일 빠르다
        - 이 기록에 가까울수록 최적화가 잘 되었다.
        - 이 기록을 추월했다면, 뭔가 잘못되었다.(단순 copy보다 copy+operation이 더 빠를 수 없다.)
    - CPU Version 
        - `memcpy(void *dest, const void *source, size_t num )` 
        - size는 byte 단위
        - dest와 source가 겹치는 애매한 경우 `memmove()` 활용(속도 엄청 떨어짐)
            ```cpp
            memcpy(matC, matA, matsize*matsize*sizeof(float));
            ```
        - 경우에 따라선(cache 상황, compiler optimize 등) memcpy보다 for loop가 더 빠를수도 있음
    - CUDA Version
        ```cpp
        __global__ void kernelMatCpy( float* C, const float* A, unsigned matsize, size_t pitch_in_elem ) {
            __shared__ float s_mat[32][32];
            register unsigned gy = blockIdx.y * blockDim.y + threadIdx.y;
            if (gy < matsize) {
                register unsigned gx = blockIdx.x * blockDim.x + threadIdx.x;
                if (gx < matsize) {
                    register unsigned idx = gy * pitch_in_elem + gx; // in element
                    s_mat[threadIdx.y][threadIdx.x] = A[idx];
                    __syncthreads();
                    C[idx] = s_mat[threadIdx.y][threadIdx.x];
                }
            }
        }
        ```
        - global memory -> shared memory -> global memory
            - copy에선 shared memory 쓸 필요 없지만, 다른 행렬 연산 시에는 압도적으로 빨라짐
        - Tile은 정사각형 모양을 선호
            ```cpp
            __shared__ float s_mat[32][32];
            ```
            - thread블록의 최대 thread 수가 1024(32x32)개
            - old code에서는 최대 thread 수가 512이었기에 (16x16), (32x16) 이기도
            - 미래에는 또 바뀔 수도 있다.

<br>

---

- Matrix Transpose
    - 전치 행렬
    - [C<sub>i,j</sub>] = [A<sub>j,i</sub>]
    - `A[y][x] = A[y*WIDTH+y]`
    - `C[y][x] = A[x][y] = A[x*WIDTH+y]`

<br>

- Shared Memory Bank Handling
    - shared memory는 SPRAM (scratch-pad RAM)
    - 물리적으로 32개(SP 수)의 4byte씩 분리된 공간으로 __banked__ 되어 있다
        <table style=text-align:center>
                <tr>
                    <td>float a[]</td>
                    <td>bank00</td>
                    <td>bank01</td>
                    <td>bank02</td>
                    <td>...</td>
                    <td>bank31</td>
                </tr>
                <tr>
                    <td>16KB<sub>128층</sub></td>
                    <td>a[0]<br>a[32]<br>a[64]<br>...</td>
                    <td>a[1]<br>a[33]<br>a[65]<br>...</td>
                    <td>a[2]<br>a[34]<br>a[66]<br>...</td>
                    <td>...</td>
                    <td>a[31]<br>a[63]<br>a[95]<br>...</td>
                </tr>
                <tr>
                    <td>32KB</td>
                    <td>...</td>
                    <td>...</td>
                    <td>...</td>
                    <td>...</td>
                    <td>...</td>
                </tr>
                <tr>
                    <td>48KB</td>
                    <td>...</td>
                    <td>...</td>
                    <td>...</td>
                    <td>...</td>
                    <td>...</td>
                </tr>
        </table>
    - 각 bank는 독립적인 단위로, 각각의 data bus로 SP와 연결된다
    - 각 쓰레드에 각 뱅크가 달라붙어서 데이터를 보내주면 1 cycle에 실행된다.
    - 1개의 bank가 broadcast도 가능. 역시 1 cycle
    - 각 쓰레드가 0, 32, 64 ... 같은 bank에 계속 접근할 경우, worst case. 32 cycle
    - Bank Conflict 가 없는 경우, 최선의 상황 기대 가능. (접근 순서는 중구난방이어도 됨)

<br>

- Matrix Transpose Bank Conflict 해결
    - 메모리 공간을 하나 어긋나게 잡아서, 낭비하는 방법으로 해결
    - `mat[32][32]` -> `mat[32][32+1]`
    - `mat[i][ty]`가 모두 같은 bank에서 모두 다른 bank로 분산된다.
        ```cpp
        __global__ void kernelMatTranspose( float* C, const float* A, unsigned matsize, size_t pitch_in_elem ) {
            __shared__ float mat[32][32 + 1];
            // pick up for the shared memory
            register unsigned gy = blockIdx.y * blockDim.y + threadIdx.y;
            register unsigned gx = blockIdx.x * blockDim.x + threadIdx.x;
            if (gy < matsize && gx < matsize) {
                register unsigned idxA = gy * pitch_in_elem + gx;
                mat[threadIdx.y][threadIdx.x] = A[idxA];
            }
            __syncthreads();
            // transposed position
            gy = blockIdx.x * blockDim.x + threadIdx.y;
            gx = blockIdx.y * blockDim.y + threadIdx.x;
            if (gy < matsize && gx < matsize) {
                register unsigned idxC = gy * pitch_in_elem + gx;
                C[idxC] = mat[threadIdx.x][threadIdx.y];
            }
        }
        ```

<br>

---

- Matrix Multiplication
    - 행렬 곱하기
    - C<sub>ij</sub> = dot product of a<sub>i_</sub> and b<sub>_j</sub>
    - <img src="https://render.githubusercontent.com/render/math?math=C_{yx} = \sum_{k=0}^{k} A_{yk} \cdot B_{kx}">
    - 위 수식 그대로, 3중 for loop가 일반적. O(n<sup>3</sup>). 너무 느려!!!
        ```cpp
        for (register unsigned y = 0; y < matsize; ++y) {
            for (register unsigned x = 0; x < matsize; ++x) {
                unsigned indC = y * matsize + x; // convert to 1D index
                register float ans = 0.0f;
                for (register unsigned k = 0; k < matsize; ++k) {
                    unsigned indA = y * matsize + k; // convert to 1D index
                    unsigned indB = k * matsize + x; // convert to 1D index
                    ans += matA[indA] * matB[indB];
                }
                matC[indC] = ans;
            }
        }
        ```
        - 다음 row의 원소를 가져올 때, 물리적으로 매우 떨어져 있음.
            -> cache miss!
    - cache miss 최소화(outer k version, loop nest opt)
        ```cpp
        memset( matC, 0, matsize * matsize * sizeof(float) );
        for (register unsigned k = 0; k < matsize; ++k) {
            for (register unsigned y = 0; y < matsize; ++y) {
                for (register unsigned x = 0; x < matsize; ++x) {
                    unsigned indC = y * matsize + x; // convert to 1D index
                    unsigned indA = y * matsize + k; // convert to 1D index
                    unsigned indB = k * matsize + x; // convert to 1D index
                    matC[indC] += matA[indA] * matB[indB];
                }
            }
        }
        ```
        - k loop를 가장 바깥으로, 전체 원소를 차곡차곡 연산하는 형태
        - 계산량은 같지만, cache를 잘 쓸 수 있어 빨라짐
    - CUDA version (shared memory X)
        ```cpp
        __global__ void kernelMatMul( float* C, const float* A, const float* B, unsigned matsize, size_t pitch_in_elem ) {
            register unsigned gy = blockIdx.y * blockDim.y + threadIdx.y; // CUDA-provided index
            register unsigned gx = blockIdx.x * blockDim.x + threadIdx.x; // CUDA-provided index
            if (gy < matsize && gx < matsize) {
                float sum = 0.0f;
                for (register unsigned k = 0; k < matsize; ++k) {
                    register unsigned idxA = gy * pitch_in_elem + k;
                    register unsigned idxB = k * pitch_in_elem + gx;
                    sum += A[idxA] * B[idxB];
                }
                register unsigned idxC = gy * pitch_in_elem + gx;
                C[idxC] = sum;
            }
        }
        ```
        - 단순히 global memory에 계속 접근하며 더하는 형태
    - CUDA version (shared memory O)
        ```cpp
        __global__ void kernelMatMul( float* C, const float* A, const float* B, unsigned matsize, size_t pitch_in_elem ) {
            __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
            __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];
            register unsigned ntiles = matsize / TILE_WIDTH;
            register unsigned gy = blockIdx.y * blockDim.y + threadIdx.y; // y-coord
            register unsigned gx = blockIdx.x * blockDim.x + threadIdx.x; // x-coord
            register float sum = 0.0f;
            for (register unsigned tile = 0; tile < ntiles; ++tile) {
                register unsigned idxA = gy * pitch_in_elem + (tile * TILE_WIDTH + threadIdx.x);
                s_A[threadIdx.y][threadIdx.x] = A[idxA];
                register unsigned idxB = (tile * TILE_WIDTH + threadIdx.y) * pitch_in_elem + gx;
                s_B[threadIdx.y][threadIdx.x] = B[idxB];
                __syncthreads();
                for (register unsigned k = 0; k < TILE_WIDTH; ++k) {
                    sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
                }
                __syncthreads();
            }
            register unsigned idxC = gy * pitch_in_elem + gx;
            C[idxC] = sum;
        }
        ```
        - Tile 형태로 분해하여 차곡 차곡 계산하는 형태로 접근
            1. 각 쓰레드가 연산 대상 tile을 가져오고 syncthread
            2. 각 쓰레드가 가져온 tile의 각 원소에 대한 부분 합 계산
            3. 각 쓰레드가 다음 연산 대상 tile을 가져오기 전후 syncthread
            4. 위 과정을 연산 대상 tile들에 대해 반복
        - shared memory를 사용하면 확실하게 빨라진다.
        - 단, Tile을 가져올 때 row-major로 가져오도록 짜는 것에 유의!
    - Tile, Matrix size Handling
        ```cpp
        __global__ void kernelMatMul( float* C, const float* A, const float* B,
                              unsigned matsize, size_t pitch_in_elem, unsigned TILE_WIDTH ) {
            // c[y][x] = sum_k a[y][k] * b[k][x]
            // c[y * WIDTH + x] = sum_k a[y*WIDTH + k] * b[k*WIDTH + x]
            __shared__ float s_A[MAX_TILE_WIDTH][MAX_TILE_WIDTH];
            __shared__ float s_B[MAX_TILE_WIDTH][MAX_TILE_WIDTH];
            register unsigned ntiles = (matsize + TILE_WIDTH - 1) / TILE_WIDTH;
            register unsigned remaining = matsize; // remained elements to be multiplied
            register unsigned gy = blockIdx.y * blockDim.y + threadIdx.y; // y-coord
            register unsigned gx = blockIdx.x * blockDim.x + threadIdx.x; // x-coord
            register float sum = 0.0f;
            for (register unsigned tile = 0; tile < ntiles; ++tile) {
                register unsigned nelem = min( remaining, TILE_WIDTH );
                remaining -= TILE_WIDTH;
                if (gy < matsize && threadIdx.x < nelem) {
                    register unsigned idxA = gy * pitch_in_elem + (tile * TILE_WIDTH + threadIdx.x);
                    s_A[threadIdx.y][threadIdx.x] = A[idxA];
                }
                if (gx < matsize && threadIdx.y < nelem) {
                    register unsigned idxB = (tile * TILE_WIDTH + threadIdx.y) * pitch_in_elem + gx;
                    s_B[threadIdx.y][threadIdx.x] = B[idxB];
                }
                __syncthreads();
                for (register unsigned k = 0; k < nelem; ++k) {
                    sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
                }
                __syncthreads();
            }
            if (gy < matsize && gx < matsize) {
                register unsigned idxC = gy * pitch_in_elem + gx;
                C[idxC] = sum;
            }
        }
        ```
        - remaining을 타일 연산 후 슬금슬금 줄이면서 마지막 타일 제어
        - TILE_WIDTH 을 인자로 받아 custom tile size에 대응
        - if문과 연산을 통한 핸들링이므로, 당연히 연산 속도는 조금 느려짐

<br>

---

- GEMM ( General Matrix-to-Matrix Multiplication )
    - Z = α AB + β C
    - A, B mat mul해서 α 곱한 후 C에 β 곱해서 더한다.
    - 특별한 것은 없고 Matrix Multiplication에 일부 연산 추가

<br>

---

- speed
    - CUDA는 아직도 발전중
    - const, register, shared, local, global
    - constant Memory의 특징
        - constant-cache에 존재
        - warp내 모든 thread가 같은 constant를 사용해야 가속 효과
    - Shared Memory와 L1 Cache
        - L1의 일부이면서(물리적으로 같은 공간), shared memory로 활용한다.
        - 일부는 사용자가 관리하고, 나머지는 CUDA kernel이 관리
        - shared memory를 많이 쓰면, L1 cache 용량 축소되어 오히려 느려질수도
        - `cudaFuncSetCacheConfig` : 커널함수별로 메모리 사이즈 조정 가능
    - Host건 Device건 메모리 클리어 할 때는 memset을 쓰자

<br>

---

- Floating Point Numbers (IEEE 754)
    - sign(부호) + exponent(지수) + mantissa(크기)
    - 1 + 8 + 23 = 32bit
    - 빠르게 최적화~


<br>

---

- Control Flow

<br>

---

- Reduction Problem
    - input values들을 summarize하여 1개의 지표로 제시
    - sum, prod, max, min, average, standard deviation 등
    - sequential 하게 구현 -> single for loop -> O(n)
    - sequential reduction의 개선책? 토너먼트!
    - Parallel Reduction Tree Algorithm
        - number of operations: O(n)
        - number of steps: log<sub>2</sub>n
            - 이론상, 1million data = 2<sup>20</sup>data -> 20steps (단, 무제한 core 필요)
        - __work-efficient__ parallel algorithm
            - sequential version에 비해 추가 작업 없이 효과적으로 처리
            - 더 많은 연산을 하지만 시간을 줄이는 알고리즘도 있음
        

- Reduction
    - warp를 빨리 종료시키면 속도가 빨라진다.
    - (변태같다)

- Volatile

- GEMV(generalized matrix-vector multiplication)
    - Z = α AX + β Y
    - Matrix transpose 해서 쓰는 것 까지는 좋다.

- Linear Search
    - 보통은 for loop로 O(n) 알고리즘.
    - 속도 차이는 크지 않다.
    - 단 CUDA 버전은 모든 원소를 다 확인했다는 차이가 있다.

- Search all occurrences
    - 모든 원소를 다 찾아라!

- Binary Search
    - 병렬 바이너리 서치?
    - 거의 안쓰인다. 적합하지 않다. STL이 훨씬 낫다.

- sort!
    - 우리강 원하는 순서로 가지런하게 둔다
    - internal sort: fits in memory
    - external sort: uses auxiliary storage
    - comparision based sorting: 비교 기반
        - quick sort, merge sort
    - non-comparison based sort: 데이터의 특별한 특징 활용
        - bucket sort, radix sort
    - 소팅을 병렬 처리로 전환할 때, shared memory를 쓰냐 gloabl memroy를 쓰냐가 중요.
    - 비교를 통해 소팅을 하다 보면, TB 안에 있는 것을 어떻게 처리하고 합치느냐?
    - 1개 쓰레드가 몇개 element를 다룰 것인가?
    - 1개만 다루는 경우 - comapre and exchange.
        - 두 데이터를 비교하려면, 둘을 주고받아서 처리.
        - 어느 한 쪽은 작은걸 가져가고, 어느 한쪽은 큰걸 가져간다.
    - 쓰레드들이 여러 element를 가지고 있다가 비슷한 일 -compare and split
        - 양쪽이 각각이 가진걸 다 주고받고, 정렬해서 반반 가져간다.
        - block 단위로 sort하는것이 sorting 알고리즘의 기본적인 관점.
    - a therad block -> 1024 theads.

- 바이토닉 소트
    - monotonically -> 일관되게 한쪽 방향으로 가는, tone이 하나밖에 없는
    - bitonic sequence -> 증가 후 감소or 감소 후 증가. tone이 2개.
    - 바이토닉 소트의 기본적인 아이디어
        - 큰 문제가 있을 때, 이를 바이토닉 스퀀스로 만듦
        - 한쪽은 미니멈, 한쪽으 맥시멈을 택함
        - -> 절반 크기의 두 바이토닉 시퀀스로 나뉘어 진다.
        - 각각을 따로 sort 한 다음 둘을 combine!

- Counting Merge Sort
    - Merge sort: 반씩 나눠서 합친다. O(nlogn). divide and conquer
    - 각 쓰레드가 1개 element의 마지막 merge된 결과의 저장되는 위치를 계산하여 그 위치에 element를 잡아넣음
    - 
