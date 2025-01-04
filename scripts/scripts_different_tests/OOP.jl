
using BenchmarkTools

module Animals
using ObjectOriented

export Animal, Snake, my_snake_ckeck, ABC

@oodef mutable struct Animal
    name :: String
    function new(theName::String)
        @mk begin
            name = theName
        end
    end

    function move(self, distanceInMeters::Number = 0)
        println("$(self.name) moved $(distanceInMeters)")
    end
end


struct SnakeNormal{T}
    x::T
end

@oodef struct Snake{T}
    x::T

    function new(x::T)
        @mk begin
            x = x
        end
    end

    function _protected_method(self)
        println("Protected method")
    end

    # dfsfdsfsd
    function snake_check(self)
        # dfdsfdsfsd
        rt = rand(1:10)
        self._protected_method()
        return rt * self.x
    end
end

# # Now define a module-level wrapper that the linter will see:
# """
# Some docstirng
# """
# function snake_check(s::Snake{T}) where {T}
#     # just forward to the inner method
#     s.snake_check()
# end

struct ABC
    sn::Snake
end

function my_snake_ckeck(s::Snake)
    rt = rand(1:10)
    return rt * s.x
end

end

using .Animals

function test()

    snake = Animals.Snake(2321.09)
    ds = snake.snake_check()
    fd = snake.move(12)

    abc = Animals.ABC(snake)
    abc.sn.snake_check()
    sn::Animals.Snake = abc.sn
    dffd = sn.snake_check()

    sn2 = Animals.Snake(21.9)
    df = sn2.snake_check()
    Animals.my_snake_ckeck(snake)
    b = @benchmark Animals.my_snake_ckeck($snake)
    display(b)

    b = @benchmark $snake.snake_check()
    display(b)
end

function test2() 
    b = @benchmark Animals.Snake(21.9)
    display(b)

    b = @benchmark Animals.SnakeNormal(21.9)
    display(b)
end

# test()
test2()