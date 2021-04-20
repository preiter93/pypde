
module tridiagonal

   implicit none

contains

subroutine solve_twodia_1d(d,u1,x,axis,n)
! =====================================================
! Solve Ax = b, where A is a banded matrix filled on
! the main diagonal and a upper diagonal with offset 2
! This arises in Helmholtz like problems when discre-
! tized with chebyshev polynomials
!
! d: N
!     diagonal
! u1: N-2
!     Diagonal with offset -2
! x: array ndim==1
!     rhs
! axis : int 
!    (not used in 1d)
! =====================================================
    integer, intent(in)   :: n,axis
    real(8), intent(in)  :: d(n),u1(n-2)
    real(8), intent(inout):: x(n)
    integer :: i
    x(1) = x(1)/d(1)
    x(2) = x(2)/d(2)
    do i=3,n
        x(i) = (x(i) - u1(i-2)*x(i-2))/d(i)
    enddo
    return
end subroutine

subroutine solve_twodia_2d(d,u1,x,axis,n,m)
! =====================================================
! Solve Ax = b, along axis
! where A is a banded matrix filled on
! the main diagonal and a upper diagonal with offset 2
! This arises in Helmholtz like problems when discre-
! tized with chebyshev polynomials
!
! d: N
!     diagonal
! u1: N-2
!     Diagonal with offset -2
! x: array ndim==2
!     rhs
! axis: int
!     Solve along axis
! =====================================================
    integer, intent(in)   :: n,m,axis
    real(8), intent(in)  :: d(n),u1(n-2)
    real(8), intent(inout):: x(n,m)
    integer :: i,k
    if (axis==0) then
        do k=1,m
            x(1,k) = x(1,k)/d(1)
            x(2,k) = x(2,k)/d(2)
            do i=3,n
                x(i,k) = (x(i,k) - u1(i-2)*x(i-2,k))/d(i)
            enddo
        enddo
    elseif (axis==1) then
        do k=1,m
            x(k,1) = x(k,1)/d(1)
            x(k,2) = x(k,2)/d(2)
            do i=3,n
                x(k,i) = (x(k,i) - u1(i-2)*x(k,i-2))/d(i)
            enddo
        enddo
    endif
    return
end subroutine

end module