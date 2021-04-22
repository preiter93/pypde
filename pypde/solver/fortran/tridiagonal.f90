
module tridiagonal

   implicit none

contains


subroutine solve_tdma(d,u1,x,axis,n)
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
! x: array ndim==1
!     rhs
! axis : int 
!    (not used in 1d)
! =====================================================
    integer, intent(in)   :: n,axis
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
!     rhs
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

subroutine solve_fdma2d(d,u1,u2,l,b,x,n,m)
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
! b: array ndim==2
!     rhs
! x: array ndim==2
!     solution
! axis : int 
!    (not used in 1d)
! =====================================================
    integer, intent(in)   :: n,m
    real(8), intent(in)  :: d(n),u1(n-2),u2(n-4),l(n-2)
    real(8), intent(in)  :: b(n,m)
    real(8), intent(out) :: x(n,m)
    integer :: i,j

    x(:,:) = b(:,:)
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
end subroutine

end module