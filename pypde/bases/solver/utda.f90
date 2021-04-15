subroutine solve_triangular(R, b, x, n)
! =====================================================
! Solve Rx = b, where R is upper triangular
!
! R : array (n,n)
! b : array (n)
! x : array (n)
! =====================================================
    integer, intent(in)   :: n
    real(8), intent(in)   :: R(n,n),b(n)
    real(8), intent(out)  :: x(n)
    real(8):: bc(n)
    integer:: i,j

    x = 0.0
    bc(:) = b(:)
    do j=1,n
        i = n+1-j
        x(i) = bc(i)/R(i,i)
        if (i/=1) bc(1:i-1) = bc(1:i-1) - x(i)*R(1:i-1,i)
    end do
    return
end subroutine