subroutine twodma_solve(d,u1,x,n)
! =====================================================
! Solve Ax = b, where a is a banded matrix filled on
! the main diagonal and a upper diagonal with offset 2
! This arises in Helmholtz like problems when discre-
! tized with chebyshev polynomials
!
!    d: N
!        diagonal
!    u1: N-2
!        Diagonal with offset -2
!    x: array ndim==1
!        rhs
! 
! =====================================================
    integer, intent(in)   :: n
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