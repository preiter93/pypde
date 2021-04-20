module triangular

   implicit none

contains

subroutine solve_1d(R, b, x, axis, n)
! =====================================================
! Solve Rx = b, where R is upper triangular
!
! R : array (n,n)
! b : array (n)
! x : array (n)
! axis : int (not used in 1d)
! =====================================================
    integer, intent(in)   :: n,axis
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

subroutine solve_2d(R, b, x, axis, n, m)
! =====================================================
! Solve Rx = b, where R is upper triangular
! System is solved along axis.
!
! R : array (n,n)
! b : array (n,m)
! x : array (n,m)
! axis : int
!
! SLOWER THAN scipy.linalg.solve_triangular !!!!!!!
! although 1D routine is faster ...
! =====================================================
    integer, intent(in)   :: n, m, axis
    real(8), intent(in)   :: R(n,n)
    real(8), intent(in)   :: b(n,m)
    real(8), intent(out)  :: x(n,m)
    real(8):: bc(n)
    integer:: i,j,k

    if (axis==0) then
        do k=1,m
            call solve_1d(R,b(:,k),x(:,k),0,n)
        enddo
    elseif (axis==1) then
        do k=1,n
            call solve_1d(R,b(k,:),x(k,:),0,m)
        enddo
    endif
    ! if (axis==0) then
    !     do k=1,m
    !         x(:,k) = 0.0
    !         bc(:) = b(:,k)
    !         do j=1,n
    !             i = n+1-j
    !             x(i,k) = bc(i)/R(i,i)
    !             if (i/=1) bc(1:i-1) = bc(1:i-1) - x(i,k)*R(1:i-1,i)
    !         end do
    !     enddo
    ! elseif (axis==1) then
    !     do k=1,m
    !         x(k,:) = 0.0
    !         bc(:) = b(k,:)
    !         do j=1,n
    !             i = n+1-j
    !             x(k,i) = bc(i)/R(i,i)
    !             if (i/=1) bc(1:i-1) = bc(1:i-1) - x(k,i)*R(1:i-1,i)
    !         end do
    !     enddo
    ! endif
    ! return

end subroutine

end module