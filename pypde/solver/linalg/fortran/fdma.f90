subroutine solve_fdma_1d(l,d,u1,u2,x,n)
! =====================================================
! Solve Ax = b, where A
! 4-diagonal matrix with diagonals in offsets -2, 0, 2, 4
!
! The diagonals must be preprocessed before calling
! this function. This happens in python. Alternatively
! see init_fdma below.
!
! l:  N-2
!     Diagonal with offset -2
! d: N
!     diagonal with offset 0
! u1: N-2
!     Diagonal with offset +2
! u2: N-4
!     Diagonal with offset +4
! x: array ndim==1
!     rhs (in) / solution (out)
! =====================================================
    integer, intent(in)  :: n
    real(8), intent(in)  :: d(n),u1(n-2),u2(n-4),l(n-2)
    real(8), intent(inout):: x(n)
    integer :: i

    do i=3,n
        x(i) = x(i) - l(i-2)*x(i-2)
    enddo

    x(n) = x(n)/d(n)
    x(n-1) = x(n-1)/d(n-1)
    x(n-2) = (x(n-2) - u1(n-2)*x(n-0))/d(n-2)
    x(n-3) = (x(n-3) - u1(n-3)*x(n-1))/d(n-3)
    do i=n-4,1,-1
        x(i) = (x(i) - u1(i)*x(i+2) - u2(i)*x(i+4))/d(i)
    enddo
    return
end subroutine

subroutine solve_fdma_2d(l,d,u1,u2,x,axis,n,m)
! =====================================================
! Solve Ax = b, where A
! 4-diagonal matrix with diagonals in offsets -2, 0, 2, 4
! and b is two dimensional (n,m)
!
! The diagonals must be preprocessed before calling
! this function. This happens in python. Alternatively
! see init_fdma below.
!
! l:  N-2
!     Diagonal with offset -2
! d: N
!     diagonal with offset 0
! u1: N-2
!     Diagonal with offset +2
! u2: N-4
!     Diagonal with offset +4
! x: array ndim==2
!     rhs (in) / solution (out)
! axis : int
!    Axis over which to solve
! =====================================================
    integer, intent(in)  :: axis,n,m
    real(8), intent(in)  :: d(:),u1(:),u2(:),l(:)
    real(8), intent(inout) :: x(n,m)
    integer :: i

    ! ------- axis 0 ------------
    if (axis==0) then
        do i=3,n
            x(i,:) = x(i,:) - l(i-2)*x(i-2,:)
        enddo

        x(n,:) = x(n,:)/d(n)
        x(n-1,:) = x(n-1,:)/d(n-1)
        x(n-2,:) = (x(n-2,:) - u1(n-2)*x(n-0,:))/d(n-2)
        x(n-3,:) = (x(n-3,:) - u1(n-3)*x(n-1,:))/d(n-3)
        do i=n-4,1,-1
            x(i,:) = (x(i,:) - u1(i)*x(i+2,:) - u2(i)*x(i+4,:))/d(i)
        enddo
        return
    ! ------- axis 1 ------------
    elseif (axis==1) then
        do i=3,m
            x(:,i) = x(:,i) - l(i-2)*x(:,i-2)
        enddo

        x(:,m) = x(:,m)/d(m)
        x(:,m-1) = x(:,m-1)/d(m-1)
        x(:,m-2) = (x(:,m-2) - u1(m-2)*x(:,m-0))/d(m-2)
        x(:,m-3) = (x(:,m-3) - u1(m-3)*x(:,m-1))/d(m-3)
        do i=m-4,1,-1
            x(:,i) = (x(:,i) - u1(i)*x(:,i+2) - u2(i)*x(:,i+4))/d(i)
        enddo
        return
    endif

end subroutine



subroutine init_fdma(A,d,u1,u2,l,n)
! =====================================================
! Call before solve_fdma, extracts diagonals and
! transforms then
!
! Input
!   A: NxN Matrix
! Output
!   d: N
!     diagonal
!   u1: N-2
!     Diagonal with offset +2
!   u2: N-4
!     Diagonal with offset +4
!    l:  N-2
!     Diagonal with offset -2
! =====================================================
integer, intent(in) :: n
real(8), intent(in) :: A(n,n)
real(8), intent(out):: d(n),u1(n-2),u2(n-4),l(n-2)
integer:: i

d = 0
u1 = 0
u2 = 0
l = 0
do i=1,n
    d(i) = A(i,i)
    if (i>2) l(i-2) = A(i,i-2)
    if (i<n-2) u1(i) = A(i,i+2)
    if (i<n-4) u2(i) = A(i,i+4)
enddo

do i=3,n
    l(i-2) = l(i-2)/d(i-2)
    d(i) = d(i) - l(i-2)*u1(i-2)
    if (i<n-2) then
        u1(i) = u1(i) - l(i-2)*u2(i-2)
    endif
enddo

end subroutine
