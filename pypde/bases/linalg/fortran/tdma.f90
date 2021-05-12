
subroutine solve_tdma_1d(a,b,c,d,k,x,n)
! =====================================================
! Tridiagonal matrix solver to solve
!          Ax = d
!  where A is banded with diagonals in offsets -k, 0, k
!
! a: n-k
!     Diagonal with offset -k
! b: n
!     diagonal
! c: n-k
!     Diagonal with offset +k
! k: int
!     Offset of sub-diagonals
! d: array ndim==1
!     rhs
! x: array ndim==1
!     solution vector (out)
! =====================================================
    integer, intent(in)   :: n
    real(8), intent(in)   :: a(:),b(:),c(:)
    integer, intent(in)   :: k
    real(8), intent(in) :: d(n)
    real(8), intent(out):: x(n)
    real(8) :: w(n-k), g(n)
    integer :: i

    ! Forward sweep
    do i=1,n-k
      if (i<k+1) then
        w(i) = c(i)/b(i)
      else
        w(i) = c(i)/(b(i) - a(i-k)*w(i-k))
      endif
    enddo

    do i=1,n
      if (i<k+1) then
        g(i) = d(i)/b(i)
      else
        g(i) = (d(i) - a(i-k)*g(i-k))/(b(i) - a(i-k)*w(i-k))
      endif
    enddo

    ! Back Substitution
    x(n-k:n) = g(n-k:n)
    do i=n-k+1,2,-1
      x(i-1) = g(i-1) - w(i-1)*x(i+k-1)
    enddo

    return
end subroutine


subroutine solve_tdma_2d(a,b,c,d,k,x,n,m)
! =====================================================
! Tridiagonal matrix solver to solve
!          Ax = d
!  where A is banded with diagonals in offsets -k, 0, k
!
! a: n-k
!     Diagonal with offset -k
! b: n
!     diagonal
! c: n-k
!     Diagonal with offset +k
! k: int
!     Offset of sub-diagonals
! d: array n x m
!     rhs
! x: array n x m
!     solution vector (out)
! =====================================================
    integer, intent(in)   :: n
    real(8), intent(in)   :: a(:),b(:),c(:)
    integer, intent(in)   :: k
    real(8), intent(in) :: d(n,m)
    real(8), intent(out):: x(n,m)
    real(8) :: w(n-k), g(n,m)
    integer :: i

    ! Forward sweep
    do i=1,n-k
      if (i<k+1) then
        w(i) = c(i)/b(i)
      else
        w(i) = c(i)/(b(i) - a(i-k)*w(i-k))
      endif
    enddo

    do i=1,n
      if (i<k+1) then
        g(i,:) = d(i,:)/b(i)
      else
        g(i,:) = (d(i,:) - a(i-k)*g(i-k,:))/(b(i) - a(i-k)*w(i-k))
      endif
    enddo

    ! Back Substitution
    x(n-k:n,:) = g(n-k:n,:)
    do i=n-k+1,2,-1
      x(i-1,:) = g(i-1,:) - w(i-1)*x(i+k-1,:)
    enddo

    return
end subroutine
