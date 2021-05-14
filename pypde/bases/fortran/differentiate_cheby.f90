subroutine diff_1d(c,dc, n)
! =====================================================
! Chebyshev differentiation:
! d_x(T_n) / n = 2 T_(n-1) + d_x(T_(n-2)) / (n-2)
!
! Calculated via recursion
!
! c: n
!    array of chebyshev coefficients
! dc: n
!    Chebyshev coefficients of first derivative
! =====================================================
    integer, intent(in)   :: n
    real(8), intent(in) :: c(n)
    real(8), intent(out):: dc(n)
    integer :: i

    dc(n-1) = 2*(n-1)*c(n)
    do i=n-2,2,-1
      dc(i) = dc(i+2) + 2*i*c(i+1)
    enddo
    dc(1) =  dc(3)/2. + c(2)

    return

end subroutine

subroutine diff_2d(c,dc, n, m)
! =====================================================
! Chebyshev differentiation:
! d_x(T_n) / n = 2 T_(n-1) + d_x(T_(n-2)) / (n-2)
!
! Calculated via recursion across first axis
!
! c: n x m
!    array of chebyshev coefficients
! dc: n x m
!    Chebyshev coefficients of first derivative
! =====================================================
    integer, intent(in)   :: n
    real(8), intent(in) :: c(n,m)
    real(8), intent(out):: dc(n,m)
    integer :: i

    dc(n-1,:) = 2*(n-1)*c(n,:)
    do i=n-2,2,-1
      dc(i,:) = dc(i+2,:) + 2*i*c(i+1,:)
    enddo
    dc(1,:) =  dc(3,:)/2. + c(2,:)

    return

end subroutine
