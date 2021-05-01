
subroutine solve_twodma_1d(d,u,x,n)
! =====================================================
! Solve Ax = b, where A is a banded matrix filled on
! diagonals in offsets 0, 2
! This arises in Poisson problems that are preconditioned
! with the pseudoinverse of D2 (chebyshev)
!
! d: N
!     diagonal
! u: N-2
!     Diagonal with offset +2
! x: array ndim==1
!     rhs (in) / solution (out)
! =====================================================
    integer, intent(in)   :: n
    real(8), intent(in)   :: d(n),u(n-2)
    real(8), intent(inout):: x(n)
    integer :: i
    x(n) = x(n)/d(n)
    x(n-1) = x(n-1)/d(n-1)
    do i=n-2,1,-1
        x(i) = (x(i) - u(i)*x(i+2))/d(i)
    enddo
    return
end subroutine

subroutine solve_twodma_2d(d,u1,x,axis,n,m)
! =====================================================
! 2D Version of solve_twodma_1d
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
        do i=n-2,1,-1
            x(i,:) = (x(i,:) - u1(i)*x(i+2,:))/d(i)
        enddo
        return
    ! ------- axis 1 ------------
    elseif (axis==1) then
        x(:,m) = x(:,m)/d(m)
        x(:,m-1) = x(:,m-1)/d(m-1)
        do i=m-2,1,-1
            x(:,i) = (x(:,i) - u1(i)*x(:,i+2))/d(i)
        enddo
        return
    endif
end subroutine
