
module tridiagonal

   implicit none

contains


subroutine solve_tdma(d,u1,x,n)
! =====================================================
! Solve Ax = b, where A is a banded matrix filled on
! diagonals in offsets 0, 2
! This arises in Poisson problems that are preconditioned
! with the pseudoinverse of D2
!
! d: N
!     diagonal
! u1: N-2
!     Diagonal with offset +2
! x: array ndim==1
!     rhs
! =====================================================
    integer, intent(in)   :: n
    real(8), intent(in)  :: d(n),u1(n-2)
    real(8), intent(inout):: x(n)
    integer :: i
    x(n) = x(n)/d(n)
    x(n-1) = x(n-1)/d(n-1)
    do i=n-3,1,-1
        x(i) = (x(i) - u1(i)*x(i+2))/d(i)
    enddo
    return
end subroutine

subroutine solve_tdma2d(d,u1,x,axis,n,m)
! =====================================================
! Solve Ax = b, where A is a banded matrix filled on
! the main diagonal and a upper diagonal with offset 2
! This arises in Poisson problems that are preconditioned
! with the pseudoinverse of D2
!
! d: N
!     diagonal
! u1: N-2
!     Diagonal with offset +2
! x: array ndim==2
!     rhs (in) / solution (out)
! axis : int 
!    Axis over which to solve
! =====================================================
    integer, intent(in)   :: n,m,axis
    real(8), intent(in)  :: d(:),u1(:)
    real(8), intent(inout):: x(n,m)
    integer :: i
    ! ------- axis 0 ------------
    if (axis==0) then
        x(n,:) = x(n,:)/d(n)
        x(n-1,:) = x(n-1,:)/d(n-1)
        do i=n-3,1,-1
            x(i,:) = (x(i,:) - u1(i)*x(i+2,:))/d(i)
        enddo
        return
    ! ------- axis 1 ------------
    elseif (axis==1) then
        x(:,m) = x(:,m)/d(m)
        x(:,m-1) = x(:,m-1)/d(m-1)
        do i=m-3,1,-1
            x(:,i) = (x(:,i) - u1(i)*x(:,i+2))/d(i)
        enddo
        return
    endif
end subroutine



subroutine solve_fdma(d,u1,u2,l,x,n)
! =====================================================
! Solve Ax = b, where A 
! 4-diagonal matrix with diagonals in offsets -2, 0, 2, 4
!
! d: N
!     diagonal
! u1: N-2
!     Diagonal with offset +2
! u2: N-4
!     Diagonal with offset +4
! l:  N-2
!     Diagonal with offset -2
! x: array ndim==1
!     rhs (in) / solution (out)
! =====================================================
    integer, intent(in)   :: n
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


subroutine solve_fdma2d(d,u1,u2,l,x,axis,n,m)
! =====================================================
! Solve Ax = b, where A 
! 4-diagonal matrix with diagonals in offsets -2, 0, 2, 4
! and b is two dimensional (n,m)
!
! d: N
!     diagonal
! u1: N-2
!     Diagonal with offset +2
! u2: N-4
!     Diagonal with offset +4
! l:  N-2
!     Diagonal with offset -2
! x: array ndim==2
!     rhs (in) / solution (out)
! axis : int 
!    Axis over which to solve
! =====================================================
    integer, intent(in)   :: axis,n,m
    real(8), intent(in)  :: d(n),u1(n-2),u2(n-4),l(n-2)
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

subroutine solve_fdma2d_type2(A,C,lam,x,axis,n,m)
! =====================================================
! Solve (lam*A + C)x = b, where LHS is a 
! 4-diagonal matrix with diagonals in offsets -2, 0, 2, 4
! and b is two dimensional (n,m)
! and lambda is a vecor which is multiplied along the
! diagonal of A
!
! A,C: NxN
!     Matrix
! lam: M
!     Array
! x: array ndim==2
!     rhs (in) / solution (out)
! axis : int 
!    Axis over which to solve
! =====================================================
    integer, intent(in)   :: axis,n,m
    real(8), intent(in)  :: A(:,:),C(:,:),lam(:)
    real(8), intent(inout) :: x(n,m)
    real(8):: d(n),u1(n-2),u2(n-4),l(n-2)
    integer :: i

    ! ------- axis 0 ------------
    if (axis==0) then
        do i=1,m
            call init_fdma( (A*lam(i) + C ), d,u1,u2,l,n)
            call solve_fdma(d,u1,u2,l,x(:,i),n)
        enddo
        return
    ! ------- axis 1 ------------
    elseif (axis==1) then
        do i=1,n
            call init_fdma( (A*lam(i) + C ), d,u1,u2,l,m)
            call solve_fdma(d,u1,u2,l,x(i,:),m)
        enddo
        return
    endif

end subroutine


end module

! subroutine solve_fdma2d(d,u1,u2,l,b,x,n,m)
! ! =====================================================
! ! Solve Ax = b, where A 
! ! 4-diagonal matrix with diagonals in offsets -2, 0, 2, 4
! ! and b is two dimensional (n,m)
! !
! ! d: N
! !     diagonal
! ! u1: N-2
! !     Diagonal with offset +2
! ! u2: N-4
! !     Diagonal with offset +4
! ! l:  N-2
! !     Diagonal with offset -2
! ! b: array ndim==2
! !     rhs
! ! x: array ndim==2
! !     solution
! ! axis : int 
! !    (not used in 1d)
! ! =====================================================
!     integer, intent(in)   :: n,m
!     real(8), intent(in)  :: d(n),u1(n-2),u2(n-4),l(n-2)
!     real(8), intent(in)  :: b(n,m)
!     real(8), intent(out) :: x(n,m)
!     integer :: i

!     x(:,:) = b(:,:)
!     do i=3,n
!         x(i,:) = x(i,:) - l(i-2)*x(i-2,:)
!     enddo

!     x(n,:) = x(n,:)/d(n)
!     x(n-1,:) = x(n-1,:)/d(n-1)
!     x(n-2,:) = (x(n-2,:) - u1(n-2)*x(n-0,:))/d(n-2)
!     x(n-3,:) = (x(n-3,:) - u1(n-3)*x(n-1,:))/d(n-3)
!     do i=n-4,1,-1
!         x(i,:) = (x(i,:) - u1(i)*x(i+2,:) - u2(i)*x(i+4,:))/d(i)
!     enddo
!     return
! end subroutine



! subroutine solve_fdma2d(d,u1,u2,l,b,x,n,m)
! ! =====================================================
! ! Solve Ax = b, where A 
! ! 4-diagonal matrix with diagonals in offsets -2, 0, 2, 4
! ! and b is two dimensional (n,m)
! !
! ! d: N
! !     diagonal
! ! u1: N-2
! !     Diagonal with offset +2
! ! u2: N-4
! !     Diagonal with offset +4
! ! l:  N-2
! !     Diagonal with offset -2
! ! b: array ndim==2
! !     rhs
! ! x: array ndim==2
! !     solution
! ! axis : int 
! !    (not used in 1d)
! ! =====================================================
!     integer, intent(in)   :: n,m
!     real(8), intent(in)  :: d(n),u1(n-2),u2(n-4),l(n-2)
!     real(8), intent(in)  :: b(n,m)
!     real(8), intent(out) :: x(n,m)
!     integer :: i,j

!     do j=1,m
!         x(:,j) = b(:,j)
!         do i=3,n
!             x(i,j) = x(i,j) - l(i-2)*x(i-2,j)
!         enddo

!         x(n,j) = x(n,j)/d(n)
!         x(n-1,j) = x(n-1,j)/d(n-1)
!         x(n-2,j) = (x(n-2,j) - u1(n-2)*x(n-0,j))/d(n-2)
!         x(n-3,j) = (x(n-3,j) - u1(n-3)*x(n-1,j))/d(n-3)
!         do i=n-4,1,-1
!             x(i,j) = (x(i,j) - u1(i)*x(i+2,j) - u2(i)*x(i+4,j))/d(i)
!         enddo
!     enddo
!     return
! end subroutine