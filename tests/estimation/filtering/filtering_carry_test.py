import jax
import jax.numpy as jnp

from ss.estimation.filtering._filtering import FilteringCarry


class TestFilteringCarry:
    def test_is_a_jittable_pytree(self) -> None:
        carry = FilteringCarry(
            time=0.0,
            prior=jnp.ones((2, 3)),
        )

        @jax.jit
        def advance(carry: FilteringCarry) -> FilteringCarry:
            return FilteringCarry(
                time=carry.time + 0.1,
                prior=carry.prior + 1,
            )

        next_carry = advance(carry)

        assert jnp.allclose(next_carry.time, 0.1)
        assert jnp.array_equal(next_carry.prior, jnp.full((2, 3), 2))
